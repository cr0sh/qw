use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QwenRopePositions {
    axes: [Vec<i32>; 3],
    pub rope_delta: i32,
}

impl QwenRopePositions {
    pub(crate) fn to_mlx(&self) -> UniquePtr<MlxArray> {
        let sequence = self.axes[0].len() as i32;
        let temporal = mlxcel_core::from_slice_i32(&self.axes[0], &[1, 1, sequence]);
        let height = mlxcel_core::from_slice_i32(&self.axes[1], &[1, 1, sequence]);
        let width = mlxcel_core::from_slice_i32(&self.axes[2], &[1, 1, sequence]);
        mlxcel_core::concatenate(
            &mlxcel_core::concatenate(&temporal, &height, 0),
            &width,
            0,
        )
    }
}

pub(crate) fn compute_rope_index(
    tokens: &[i32],
    grids: &[(i32, i32, i32)],
    spatial_merge_size: usize,
    image_token_id: i32,
    video_token_id: i32,
) -> Result<QwenRopePositions> {
    ensure!(!tokens.is_empty(), "multimodal prompt is empty");
    ensure!(spatial_merge_size > 0, "spatial merge size is zero");
    let merge = spatial_merge_size as i32;
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    let mut image_index = 0;
    let mut segment_start = 0;
    let mut current_position = 0;
    let mut cursor = 0;

    while cursor < tokens.len() {
        if tokens[cursor] != image_token_id && tokens[cursor] != video_token_id {
            cursor += 1;
            continue;
        }
        let vision_start = cursor;
        while cursor < tokens.len()
            && (tokens[cursor] == image_token_id || tokens[cursor] == video_token_id)
        {
            cursor += 1;
        }
        for position in current_position..current_position + (vision_start - segment_start) as i32 {
            for axis in &mut axes {
                axis.push(position);
            }
        }
        current_position += (vision_start - segment_start) as i32;
        let &(temporal, height, width) = grids.get(image_index).ok_or_else(|| {
            anyhow::anyhow!("image token run count exceeds image grid count")
        })?;
        ensure!(
            height % merge == 0 && width % merge == 0,
            "image grid is not divisible by spatial merge size"
        );
        let merged_height = height / merge;
        let merged_width = width / merge;
        let expected = (temporal * merged_height * merged_width) as usize;
        ensure!(
            cursor - vision_start == expected,
            "image token run {image_index} has {} tokens, expected {expected}",
            cursor - vision_start
        );
        for time in 0..temporal {
            for row in 0..merged_height {
                for column in 0..merged_width {
                    axes[0].push(current_position + time);
                    axes[1].push(current_position + row);
                    axes[2].push(current_position + column);
                }
            }
        }
        current_position += temporal.max(merged_height).max(merged_width);
        image_index += 1;
        segment_start = cursor;
    }

    ensure!(
        image_index == grids.len(),
        "image grid count exceeds image token run count"
    );
    for position in current_position..current_position + (tokens.len() - segment_start) as i32 {
        for axis in &mut axes {
            axis.push(position);
        }
    }
    ensure!(
        axes.iter().all(|axis| axis.len() == tokens.len()),
        "multimodal position count does not match prompt length"
    );
    let max_position = axes
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .expect("nonempty positions");
    Ok(QwenRopePositions {
        axes,
        rope_delta: max_position + 1 - tokens.len() as i32,
    })
}

pub(crate) fn decode_rope_positions(
    cache_offset: i32,
    sequence_length: i32,
    rope_delta: i32,
) -> UniquePtr<MlxArray> {
    let positions = mlxcel_core::arange_i32(
        cache_offset + rope_delta,
        cache_offset + rope_delta + sequence_length,
        1,
    );
    mlxcel_core::broadcast_to(
        &mlxcel_core::reshape(&positions, &[1, 1, sequence_length]),
        &[3, 1, sequence_length],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_text_image_axes_and_signed_delta() {
        let positions = compute_rope_index(
            &[1, 100, 103, 103, 103, 103, 101, 9],
            &[(1, 4, 4)],
            2,
            103,
            104,
        )
        .expect("positions");
        assert_eq!(positions.axes[0], vec![0, 1, 2, 2, 2, 2, 4, 5]);
        assert_eq!(positions.axes[1], vec![0, 1, 2, 2, 3, 3, 4, 5]);
        assert_eq!(positions.axes[2], vec![0, 1, 2, 3, 2, 3, 4, 5]);
        assert_eq!(positions.rope_delta, -2);
    }

    #[test]
    fn decode_continues_from_cache_offset_plus_delta() {
        let positions = decode_rope_positions(8, 2, -2);
        mlxcel_core::eval(&positions);
        let values = mlxcel_core::array_to_raw_bytes(&positions)
            .chunks_exact(4)
            .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 byte width")))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![6, 7, 6, 7, 6, 7]);
    }

    #[test]
    fn rejects_grid_and_token_run_mismatches() {
        assert!(compute_rope_index(&[103], &[(1, 4, 4)], 2, 103, 104).is_err());
        assert!(compute_rope_index(&[103; 4], &[], 2, 103, 104).is_err());
    }
}
