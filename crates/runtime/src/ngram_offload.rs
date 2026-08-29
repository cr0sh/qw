use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use memmap2::{Advice, Mmap, MmapOptions, UncheckedAdvice};
use safetensors::SafeTensors;
use serde::Deserialize;

const MAGIC: &[u8; 8] = b"QWNG4A01";
const VERSION: u32 = 1;
const FIXED_HEADER_BYTES: usize = 32;
const SHARD_HEADER_BYTES: usize = 16;
const DATA_ALIGNMENT: usize = 4096;
const COPY_ROWS: usize = 65_536;

#[derive(Debug, Clone)]
pub(crate) struct NGramTableSpec {
    pub tensor_prefix: String,
    pub shard_count: usize,
    pub embedding_dim: usize,
    pub group_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct ShardLayout {
    rows: usize,
    data_offset: usize,
}

pub(crate) struct NGramTable {
    map: Mmap,
    shards: Vec<ShardLayout>,
    shard_ends: Vec<usize>,
    embedding_dim: usize,
    group_size: usize,
    packed_bytes: usize,
    parameter_bytes: usize,
    row_bytes: usize,
}

impl NGramTable {
    pub(crate) fn prepare(model_dir: &Path, spec: &NGramTableSpec) -> Result<Self> {
        let path = model_dir.join("ngram_table.bin");
        if path.is_file() {
            return Self::open(&path, spec);
        }
        build_table(model_dir, &path, spec)?;
        Self::open(&path, spec)
    }

    fn open(path: &Path, spec: &NGramTableSpec) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("failed to open SSD n-gram table {}", path.display()))?;
        let map = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to mmap SSD n-gram table {}", path.display()))?;
        map.advise(Advice::Random)
            .with_context(|| format!("failed to mark {} for random access", path.display()))?;
        ensure!(
            map.len() >= FIXED_HEADER_BYTES,
            "truncated n-gram table {}",
            path.display()
        );
        ensure!(
            &map[..8] == MAGIC,
            "invalid n-gram table magic in {}",
            path.display()
        );
        ensure!(
            read_u32(&map, 8)? == VERSION,
            "unsupported n-gram table version in {}",
            path.display()
        );
        let shard_count = read_u32(&map, 12)? as usize;
        let embedding_dim = read_u32(&map, 16)? as usize;
        let group_size = read_u32(&map, 20)? as usize;
        let total_rows = read_u64(&map, 24)? as usize;
        ensure!(
            shard_count == spec.shard_count,
            "n-gram table shard count mismatch: {shard_count} != {}",
            spec.shard_count
        );
        ensure!(
            embedding_dim == spec.embedding_dim,
            "n-gram table embedding dimension mismatch: {embedding_dim} != {}",
            spec.embedding_dim
        );
        ensure!(
            group_size == spec.group_size,
            "n-gram table group size mismatch: {group_size} != {}",
            spec.group_size
        );
        ensure!(
            embedding_dim.is_multiple_of(group_size),
            "n-gram embedding dimension must be divisible by group size"
        );
        ensure!(
            embedding_dim.is_multiple_of(8),
            "4-bit n-gram embedding dimension must be divisible by 8"
        );

        let packed_bytes = embedding_dim / 2;
        let parameter_bytes = embedding_dim / group_size * 2;
        let row_bytes = packed_bytes + parameter_bytes * 2;
        let mut shards = Vec::with_capacity(shard_count);
        let mut shard_ends = Vec::with_capacity(shard_count);
        let mut rows_seen = 0usize;
        for shard in 0..shard_count {
            let header = FIXED_HEADER_BYTES + shard * SHARD_HEADER_BYTES;
            let rows = read_u64(&map, header)? as usize;
            let data_offset = read_u64(&map, header + 8)? as usize;
            let data_bytes = rows
                .checked_mul(row_bytes)
                .context("n-gram shard byte size overflow")?;
            ensure!(
                data_offset
                    .checked_add(data_bytes)
                    .is_some_and(|end| end <= map.len()),
                "n-gram shard {shard} exceeds table file"
            );
            rows_seen = rows_seen
                .checked_add(rows)
                .context("n-gram row count overflow")?;
            shards.push(ShardLayout { rows, data_offset });
            shard_ends.push(rows_seen);
        }
        ensure!(
            rows_seen == total_rows,
            "n-gram table row count mismatch: {rows_seen} != {total_rows}"
        );

        Ok(Self {
            map,
            shards,
            shard_ends,
            embedding_dim,
            group_size,
            packed_bytes,
            parameter_bytes,
            row_bytes,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.shard_ends.last().copied().unwrap_or(0)
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub(crate) fn gather_bf16(&self, indices: &[usize]) -> Result<Vec<u8>> {
        let mut output = vec![0u8; indices.len() * self.embedding_dim * 2];
        let mut discard_pages = Vec::with_capacity(indices.len());
        for (output_row, &index) in output.chunks_exact_mut(self.embedding_dim * 2).zip(indices) {
            let shard_index = self.shard_ends.partition_point(|&end| end <= index);
            ensure!(
                shard_index < self.shards.len(),
                "n-gram embedding index {index} is outside {} rows",
                self.rows()
            );
            let shard_start = if shard_index == 0 {
                0
            } else {
                self.shard_ends[shard_index - 1]
            };
            let local_row = index - shard_start;
            let shard = self.shards[shard_index];
            debug_assert!(local_row < shard.rows);
            let row_start = shard.data_offset + local_row * self.row_bytes;
            let packed = &self.map[row_start..row_start + self.packed_bytes];
            let scales_start = row_start + self.packed_bytes;
            let biases_start = scales_start + self.parameter_bytes;
            let scales = &self.map[scales_start..biases_start];
            let biases = &self.map[biases_start..biases_start + self.parameter_bytes];

            for group in 0..self.embedding_dim / self.group_size {
                let scale = bf16_to_f32(u16::from_le_bytes([
                    scales[group * 2],
                    scales[group * 2 + 1],
                ]));
                let bias = bf16_to_f32(u16::from_le_bytes([
                    biases[group * 2],
                    biases[group * 2 + 1],
                ]));
                let start = group * self.group_size;
                let end = start + self.group_size;
                for dimension in start..end {
                    let byte = packed[dimension / 2];
                    let quantized = if dimension.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    let value = scale.mul_add(quantized as f32, bias);
                    output_row[dimension * 2..dimension * 2 + 2]
                        .copy_from_slice(&f32_to_bf16(value).to_le_bytes());
                }
            }
            let page_start = row_start / DATA_ALIGNMENT * DATA_ALIGNMENT;
            let page_end = align_up(row_start + self.row_bytes, DATA_ALIGNMENT).min(self.map.len());
            discard_pages.push((page_start, page_end));
        }

        // The table is a clean, separate file mapping. Release pages after the
        // selected rows have been copied so they remain pageable SSD data and
        // never accumulate alongside Metal-resident model buffers.
        discard_pages.sort_unstable();
        let mut current: Option<(usize, usize)> = None;
        for (start, end) in discard_pages {
            match current {
                Some((range_start, range_end)) if start <= range_end => {
                    current = Some((range_start, range_end.max(end)));
                }
                Some((range_start, range_end)) => {
                    // SAFETY: every row borrow ended before this loop. This is
                    // a clean shared file mapping, so future reads repopulate
                    // the original bytes from the n-gram table.
                    let _ = unsafe {
                        self.map.unchecked_advise_range(
                            UncheckedAdvice::DontNeed,
                            range_start,
                            range_end - range_start,
                        )
                    };
                    current = Some((start, end));
                }
                None => current = Some((start, end)),
            }
        }
        if let Some((start, end)) = current {
            // SAFETY: no mapping borrow survives the copy loop; see above.
            let _ = unsafe {
                self.map
                    .unchecked_advise_range(UncheckedAdvice::DontNeed, start, end - start)
            };
        }
        Ok(output)
    }
}

#[derive(Deserialize)]
struct ShardIndex {
    weight_map: BTreeMap<String, String>,
}

fn build_table(model_dir: &Path, destination: &Path, spec: &NGramTableSpec) -> Result<()> {
    ensure!(
        spec.shard_count > 0,
        "n-gram table must have at least one shard"
    );
    ensure!(
        spec.embedding_dim.is_multiple_of(spec.group_size),
        "n-gram embedding dimension must be divisible by group size"
    );
    ensure!(
        spec.embedding_dim.is_multiple_of(8),
        "4-bit n-gram embedding dimension must be divisible by 8"
    );

    let index_path = model_dir.join("model.safetensors.index.json");
    let index: ShardIndex = serde_json::from_slice(
        &fs::read(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?,
    )
    .with_context(|| format!("failed to parse {}", index_path.display()))?;

    let packed_bytes = spec.embedding_dim / 2;
    let parameter_bytes = spec.embedding_dim / spec.group_size * 2;
    let row_bytes = packed_bytes + parameter_bytes * 2;
    let data_start = align_up(
        FIXED_HEADER_BYTES + spec.shard_count * SHARD_HEADER_BYTES,
        DATA_ALIGNMENT,
    );
    let temporary = temporary_path(destination);
    let file = File::create(&temporary).with_context(|| {
        format!(
            "failed to create temporary n-gram table {}",
            temporary.display()
        )
    })?;
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    writer.write_all(&vec![0u8; data_start])?;

    let mut layouts = Vec::with_capacity(spec.shard_count);
    let mut total_rows = 0usize;
    for shard in 0..spec.shard_count {
        let base = format!("{}.{}", spec.tensor_prefix, shard);
        let weight_name = format!("{base}.weight");
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");
        let source_name = index
            .weight_map
            .get(&weight_name)
            .with_context(|| format!("missing {weight_name} from checkpoint index"))?;
        ensure!(
            index.weight_map.get(&scales_name) == Some(source_name),
            "{scales_name} is not colocated with {weight_name}"
        );
        ensure!(
            index.weight_map.get(&biases_name) == Some(source_name),
            "{biases_name} is not colocated with {weight_name}"
        );

        let source_path = model_dir.join(source_name);
        let source_file = File::open(&source_path).with_context(|| {
            format!(
                "failed to open n-gram source shard {}",
                source_path.display()
            )
        })?;
        let source_map = unsafe { MmapOptions::new().map(&source_file) }.with_context(|| {
            format!(
                "failed to mmap n-gram source shard {}",
                source_path.display()
            )
        })?;
        source_map.advise(Advice::Sequential)?;
        let tensors = SafeTensors::deserialize(&source_map).with_context(|| {
            format!(
                "failed to parse n-gram source shard {}",
                source_path.display()
            )
        })?;
        let weight = tensors.tensor(&weight_name)?;
        let scales = tensors.tensor(&scales_name)?;
        let biases = tensors.tensor(&biases_name)?;
        ensure!(
            weight.dtype() == safetensors::Dtype::U32,
            "{weight_name} must be U32"
        );
        ensure!(
            scales.dtype() == safetensors::Dtype::BF16,
            "{scales_name} must be BF16"
        );
        ensure!(
            biases.dtype() == safetensors::Dtype::BF16,
            "{biases_name} must be BF16"
        );
        ensure!(
            weight.shape().len() == 2 && weight.shape()[1] * 4 == packed_bytes,
            "unexpected {weight_name} shape {:?}",
            weight.shape()
        );
        let rows = weight.shape()[0];
        let groups = spec.embedding_dim / spec.group_size;
        ensure!(
            scales.shape() == [rows, groups],
            "unexpected {scales_name} shape {:?}",
            scales.shape()
        );
        ensure!(
            biases.shape() == [rows, groups],
            "unexpected {biases_name} shape {:?}",
            biases.shape()
        );

        let data_offset = writer.stream_position()? as usize;
        layouts.push(ShardLayout { rows, data_offset });
        let weight_data = weight.data();
        let scales_data = scales.data();
        let biases_data = biases.data();
        let mut interleaved = Vec::with_capacity(COPY_ROWS * row_bytes);
        for first_row in (0..rows).step_by(COPY_ROWS) {
            let count = (rows - first_row).min(COPY_ROWS);
            interleaved.clear();
            for row in first_row..first_row + count {
                interleaved
                    .extend_from_slice(&weight_data[row * packed_bytes..(row + 1) * packed_bytes]);
                interleaved.extend_from_slice(
                    &scales_data[row * parameter_bytes..(row + 1) * parameter_bytes],
                );
                interleaved.extend_from_slice(
                    &biases_data[row * parameter_bytes..(row + 1) * parameter_bytes],
                );
            }
            writer.write_all(&interleaved)?;
        }
        total_rows = total_rows
            .checked_add(rows)
            .context("n-gram row count overflow")?;
    }

    writer.flush()?;
    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&(spec.shard_count as u32).to_le_bytes())?;
    writer.write_all(&(spec.embedding_dim as u32).to_le_bytes())?;
    writer.write_all(&(spec.group_size as u32).to_le_bytes())?;
    writer.write_all(&(total_rows as u64).to_le_bytes())?;
    for layout in &layouts {
        writer.write_all(&(layout.rows as u64).to_le_bytes())?;
        writer.write_all(&(layout.data_offset as u64).to_le_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary, destination).with_context(|| {
        format!(
            "failed to install SSD n-gram table {}",
            destination.display()
        )
    })?;
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value: [u8; 4] = bytes
        .get(offset..offset + 4)
        .context("truncated n-gram table header")?
        .try_into()
        .expect("length checked");
    Ok(u32::from_le_bytes(value))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value: [u8; 8] = bytes
        .get(offset..offset + 8)
        .context("truncated n-gram table header")?
        .try_into()
        .expect("length checked");
    Ok(u64::from_le_bytes(value))
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn temporary_path(destination: &Path) -> PathBuf {
    destination.with_extension(format!("bin.tmp-{}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_bf16(value: f32) -> [u8; 2] {
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }

    fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()
    }

    fn write_fixture(path: &Path) {
        let data_start = DATA_ALIGNMENT;
        let rows = 2usize;
        let embedding_dim = 8usize;
        let group_size = 4usize;
        let packed_bytes = 4usize;
        let parameter_bytes = 4usize;
        let mut bytes = vec![0u8; data_start + rows * (packed_bytes + 2 * parameter_bytes)];
        bytes[..8].copy_from_slice(MAGIC);
        bytes[8..12].copy_from_slice(&VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&(embedding_dim as u32).to_le_bytes());
        bytes[20..24].copy_from_slice(&(group_size as u32).to_le_bytes());
        bytes[24..32].copy_from_slice(&(rows as u64).to_le_bytes());
        bytes[32..40].copy_from_slice(&(rows as u64).to_le_bytes());
        bytes[40..48].copy_from_slice(&(data_start as u64).to_le_bytes());

        let row_bytes = packed_bytes + 2 * parameter_bytes;
        for row in 0..rows {
            let base = data_start + row * row_bytes;
            bytes[base..base + 4].copy_from_slice(&[0x10, 0x32, 0x54, 0x76]);
            bytes[base + 4..base + 6].copy_from_slice(&f32_to_bf16(1.0 + row as f32));
            bytes[base + 6..base + 8].copy_from_slice(&f32_to_bf16(0.5));
            bytes[base + 8..base + 10].copy_from_slice(&f32_to_bf16(10.0));
            bytes[base + 10..base + 12].copy_from_slice(&f32_to_bf16(-1.0));
        }
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn gathers_and_dequantizes_only_requested_rows() {
        let directory = std::env::temp_dir().join(format!("qw-ngram-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("ngram_table.bin");
        write_fixture(&path);
        let spec = NGramTableSpec {
            tensor_prefix: "unused".into(),
            shard_count: 1,
            embedding_dim: 8,
            group_size: 4,
        };
        let table = NGramTable::open(&path, &spec).unwrap();
        assert_eq!(table.rows(), 2);
        assert_eq!(table.embedding_dim(), 8);
        assert_eq!(
            decode_bf16(&table.gather_bf16(&[1, 0]).unwrap()),
            vec![
                10.0, 12.0, 14.0, 16.0, 1.0, 1.5, 2.0, 2.5, 10.0, 11.0, 12.0, 13.0, 1.0, 1.5, 2.0,
                2.5
            ]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_out_of_range_rows() {
        let directory = std::env::temp_dir().join(format!("qw-ngram-oob-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("ngram_table.bin");
        write_fixture(&path);
        let spec = NGramTableSpec {
            tensor_prefix: "unused".into(),
            shard_count: 1,
            embedding_dim: 8,
            group_size: 4,
        };
        let table = NGramTable::open(&path, &spec).unwrap();
        assert!(
            table
                .gather_bf16(&[2])
                .unwrap_err()
                .to_string()
                .contains("outside 2 rows")
        );
        fs::remove_dir_all(directory).unwrap();
    }
}
