// SPDX-License-Identifier: Apache-2.0
//
// Minimal reader for the torch `.pt` container (a zip of STORED entries plus a
// protocol-2 pickle) as written by `torch.save(dict_of_tensors)`. It understands
// only what such a file contains: a (possibly nested-free) dict of
// `torch._utils._rebuild_tensor_v2` tensors over `FloatStorage`. Anything else is
// an error, so a hostile or unexpected file can never execute or allocate wildly.

use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{Context, Result, bail, ensure};

/// One float32 tensor read from a torch archive, row-major contiguous.
#[derive(Debug, Clone, PartialEq)]
pub struct TorchTensorF32 {
    pub shape: Vec<usize>,
    /// Little-endian f32 bytes, `shape.product() * 4` long.
    pub bytes: Vec<u8>,
}

fn u16_at(b: &[u8], at: usize) -> Result<u16> {
    let s = b.get(at..at + 2).context("truncated zip")?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32> {
    let s = b.get(at..at + 4).context("truncated zip")?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Entries of a zip archive whose members are all stored uncompressed.
fn zip_stored_entries(bytes: &[u8]) -> Result<HashMap<String, &[u8]>> {
    ensure!(bytes.len() >= 22, "not a zip archive: file too short");
    let search_start = bytes.len().saturating_sub(22 + 65535);
    let eocd = (search_start..=bytes.len() - 22)
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .context("not a torch .pt file: zip end-of-central-directory record not found")?;
    let count = u16_at(bytes, eocd + 10)? as usize;
    let cd_offset = u32_at(bytes, eocd + 16)? as usize;
    ensure!(
        count != 0xFFFF && cd_offset != 0xFFFF_FFFF,
        "zip64 archives are not supported"
    );
    let mut entries = HashMap::new();
    let mut at = cd_offset;
    for _ in 0..count {
        ensure!(
            u32_at(bytes, at)? == 0x0201_4b50,
            "corrupt zip central directory"
        );
        let method = u16_at(bytes, at + 10)?;
        let comp = u32_at(bytes, at + 20)? as usize;
        let uncomp = u32_at(bytes, at + 24)? as usize;
        let name_len = u16_at(bytes, at + 28)? as usize;
        let extra_len = u16_at(bytes, at + 30)? as usize;
        let comment_len = u16_at(bytes, at + 32)? as usize;
        let local = u32_at(bytes, at + 42)? as usize;
        let name = std::str::from_utf8(
            bytes
                .get(at + 46..at + 46 + name_len)
                .context("truncated zip")?,
        )
        .context("non-UTF-8 zip entry name")?
        .to_owned();
        at += 46 + name_len + extra_len + comment_len;
        ensure!(
            method == 0 && comp == uncomp,
            "zip entry `{name}` is compressed (method {method}); torch archives store tensors uncompressed"
        );
        ensure!(
            u32_at(bytes, local)? == 0x0403_4b50,
            "corrupt zip local header for `{name}`"
        );
        let l_name = u16_at(bytes, local + 26)? as usize;
        let l_extra = u16_at(bytes, local + 28)? as usize;
        let start = local + 30 + l_name + l_extra;
        let data = bytes
            .get(start..start + uncomp)
            .with_context(|| format!("zip entry `{name}` extends past the end of the file"))?;
        entries.insert(name, data);
    }
    Ok(entries)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum V {
    Mark,
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    /// Items and nesting depth (1 for a flat tuple); depth is capped so drops and walks stay shallow.
    Tuple(Rc<[V]>, u8),
    Global(String),
    Dict(Vec<(V, V)>),
    Storage {
        key: String,
    },
    Tensor(TensorRef),
}

#[derive(Debug, Clone)]
struct TensorRef {
    key: String,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl V {
    fn kind(&self) -> &'static str {
        match self {
            V::Mark => "mark",
            V::None => "None",
            V::Bool(_) => "bool",
            V::Int(_) => "int",
            V::Str(_) => "string",
            V::Tuple(..) => "tuple",
            V::Global(_) => "global",
            V::Dict(_) => "dict",
            V::Storage { .. } => "storage",
            V::Tensor(_) => "tensor",
        }
    }
}

const MAX_PICKLE_BYTES: usize = 1 << 20;
const MAX_TUPLE_DEPTH: u8 = 2;
const MAX_TENSOR_RANK: usize = 8;

fn make_tuple(items: Vec<V>) -> Result<V> {
    let depth = items
        .iter()
        .map(|v| if let V::Tuple(_, d) = v { *d } else { 0 })
        .max()
        .unwrap_or(0)
        + 1;
    ensure!(depth <= MAX_TUPLE_DEPTH, "pickle: tuples nested too deeply");
    Ok(V::Tuple(items.into(), depth))
}

/// Memo entries are cheap to clone (Rc tuples, scalars). Dicts and tensors are never
/// legitimately fetched back, so they are stored as an opaque marker instead of cloned.
fn memo_value(v: &V) -> Result<V> {
    Ok(match v {
        V::Dict(_) | V::Tensor(_) => V::Mark,
        other => other.clone(),
    })
}

fn as_usize(v: &V) -> Result<usize> {
    match v {
        V::Int(i) if *i >= 0 => Ok(*i as usize),
        other => bail!(
            "expected a non-negative integer in the pickle, got a {}",
            other.kind()
        ),
    }
}

fn usize_tuple(v: &V) -> Result<Vec<usize>> {
    match v {
        V::Tuple(items, _) => items.iter().map(as_usize).collect(),
        other => bail!("expected a tuple in the pickle, got a {}", other.kind()),
    }
}

fn pop_to_mark(stack: &mut Vec<V>) -> Result<Vec<V>> {
    let pos = stack
        .iter()
        .rposition(|v| matches!(v, V::Mark))
        .context("pickle: MARK not found")?;
    let items = stack.split_off(pos + 1);
    stack.pop();
    Ok(items)
}

fn memo_get(memo: &HashMap<usize, V>, k: usize) -> Result<V> {
    match memo.get(&k).context("pickle: unknown memo key")? {
        V::Mark => bail!("pickle: unsupported memo reference to a dict or tensor"),
        v => Ok(v.clone()),
    }
}

fn run_pickle(data: &[u8]) -> Result<V> {
    ensure!(
        data.len() <= MAX_PICKLE_BYTES,
        "pickle is implausibly large for a map file"
    );
    let mut stack: Vec<V> = Vec::new();
    let mut memo: HashMap<usize, V> = HashMap::new();
    let mut i = 0usize;
    macro_rules! take {
        ($n:expr) => {{
            let n = $n;
            let s = data.get(i..i + n).context("truncated pickle")?;
            i += n;
            s
        }};
    }
    macro_rules! pop {
        () => {
            stack.pop().context("pickle stack underflow")?
        };
    }
    loop {
        let op = *data.get(i).context("pickle ended without STOP")?;
        i += 1;
        match op {
            0x80 => {
                let proto = take!(1)[0];
                ensure!(
                    (2..=4).contains(&proto),
                    "unsupported pickle protocol {proto}"
                );
            }
            0x95 => {
                take!(8);
            }
            b'}' => stack.push(V::Dict(Vec::new())),
            b')' => stack.push(V::Tuple(Rc::from(Vec::new()), 1)),
            b'N' => stack.push(V::None),
            0x88 => stack.push(V::Bool(true)),
            0x89 => stack.push(V::Bool(false)),
            b'(' => stack.push(V::Mark),
            b'q' => {
                let k = take!(1)[0] as usize;
                memo.insert(
                    k,
                    memo_value(stack.last().context("pickle stack underflow")?)?,
                );
            }
            b'r' => {
                let s = take!(4);
                let k = u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
                memo.insert(
                    k,
                    memo_value(stack.last().context("pickle stack underflow")?)?,
                );
            }
            0x94 => {
                let k = memo.len();
                memo.insert(
                    k,
                    memo_value(stack.last().context("pickle stack underflow")?)?,
                );
            }
            b'h' => {
                let k = take!(1)[0] as usize;
                stack.push(memo_get(&memo, k)?);
            }
            b'j' => {
                let s = take!(4);
                let k = u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize;
                stack.push(memo_get(&memo, k)?);
            }
            b'K' => stack.push(V::Int(i64::from(take!(1)[0]))),
            b'M' => {
                let s = take!(2);
                stack.push(V::Int(i64::from(u16::from_le_bytes([s[0], s[1]]))));
            }
            b'J' => {
                let s = take!(4);
                stack.push(V::Int(i64::from(i32::from_le_bytes([
                    s[0], s[1], s[2], s[3],
                ]))));
            }
            b'X' | 0x8c => {
                let n = if op == b'X' {
                    let s = take!(4);
                    u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize
                } else {
                    take!(1)[0] as usize
                };
                let s = std::str::from_utf8(take!(n)).context("non-UTF-8 pickle string")?;
                stack.push(V::Str(s.to_owned()));
            }
            b'c' => {
                let line = |i: &mut usize| -> Result<String> {
                    let rest = data.get(*i..).context("truncated pickle")?;
                    let end = rest
                        .iter()
                        .position(|&b| b == b'\n')
                        .context("truncated pickle GLOBAL")?;
                    let s = std::str::from_utf8(&rest[..end])
                        .context("non-UTF-8 pickle GLOBAL")?
                        .to_owned();
                    *i += end + 1;
                    Ok(s)
                };
                let module = line(&mut i)?;
                let name = line(&mut i)?;
                stack.push(V::Global(format!("{module} {name}")));
            }
            b't' => {
                let items = pop_to_mark(&mut stack)?;
                stack.push(make_tuple(items)?);
            }
            0x85 | 0x86 | 0x87 => {
                let n = usize::from(op - 0x84);
                ensure!(stack.len() >= n, "pickle stack underflow");
                let items = stack.split_off(stack.len() - n);
                stack.push(make_tuple(items)?);
            }
            b'Q' => {
                let V::Tuple(pid, _) = pop!() else {
                    bail!("pickle: persistent id is not a tuple");
                };
                match &pid[..] {
                    [V::Str(tag), V::Global(g), V::Str(key), _location, _numel]
                        if tag == "storage" =>
                    {
                        ensure!(
                            g == "torch FloatStorage",
                            "unsupported torch storage type `{g}`; only float32 maps are supported"
                        );
                        stack.push(V::Storage { key: key.clone() });
                    }
                    _ => bail!("pickle: unexpected persistent id"),
                }
            }
            b'R' => {
                let args = pop!();
                let callable = pop!();
                let V::Global(name) = callable else {
                    bail!("pickle: REDUCE on a non-global");
                };
                match (name.as_str(), args) {
                    ("collections OrderedDict", V::Tuple(a, _)) if a.is_empty() => {
                        stack.push(V::Dict(Vec::new()));
                    }
                    ("torch._utils _rebuild_tensor_v2", V::Tuple(a, _)) if a.len() >= 4 => {
                        let V::Storage { key } = &a[0] else {
                            bail!("pickle: tensor without a storage");
                        };
                        let shape = usize_tuple(&a[2])?;
                        let stride = usize_tuple(&a[3])?;
                        ensure!(
                            shape.len() <= MAX_TENSOR_RANK && shape.len() == stride.len(),
                            "pickle: unsupported tensor rank"
                        );
                        stack.push(V::Tensor(TensorRef {
                            key: key.clone(),
                            offset: as_usize(&a[1])?,
                            shape,
                            stride,
                        }));
                    }
                    (other, _) => bail!("pickle: refusing to call `{other}`"),
                }
            }
            b's' => {
                let v = pop!();
                let k = pop!();
                let V::Dict(d) = stack.last_mut().context("pickle stack underflow")? else {
                    bail!("pickle: SETITEM on a non-dict");
                };
                ensure!(
                    matches!((&k, &v), (V::Str(_), V::Tensor(_))),
                    "pickle: dict entries must map strings to tensors"
                );
                d.push((k, v));
            }
            b'u' => {
                let items = pop_to_mark(&mut stack)?;
                ensure!(items.len() % 2 == 0, "pickle: odd SETITEMS");
                let V::Dict(d) = stack.last_mut().context("pickle stack underflow")? else {
                    bail!("pickle: SETITEMS on a non-dict");
                };
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    ensure!(
                        matches!((&k, &v), (V::Str(_), V::Tensor(_))),
                        "pickle: dict entries must map strings to tensors"
                    );
                    d.push((k, v));
                }
            }
            b'.' => return stack.pop().context("pickle: empty stack at STOP"),
            other => bail!("pickle: unsupported opcode 0x{other:02x}"),
        }
    }
}

/// Read every tensor of a torch `.pt` file (a flat dict of float32 tensors).
/// Tensors must be contiguous; the result keeps the dict's order.
pub fn read_f32_tensor_dict(bytes: &[u8]) -> Result<Vec<(String, TorchTensorF32)>> {
    let entries = zip_stored_entries(bytes)?;
    let pkl_name = entries
        .keys()
        .find(|n| n.ends_with("/data.pkl") || *n == "data.pkl")
        .context("not a torch .pt file: no data.pkl")?
        .clone();
    let stem = pkl_name.strip_suffix("data.pkl").unwrap_or("");
    if let Some(order) = entries.get(format!("{stem}byteorder").as_str()) {
        ensure!(
            *order == b"little",
            "big-endian torch archives are not supported"
        );
    }
    let V::Dict(items) = run_pickle(entries[&pkl_name])? else {
        bail!("torch archive does not hold a dict of tensors");
    };
    let mut out = Vec::with_capacity(items.len());
    for (k, v) in items {
        let (V::Str(name), V::Tensor(t)) = (k, v) else {
            bail!("torch archive dict must map strings to tensors");
        };
        let numel = t
            .shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .and_then(|n| n.checked_mul(4).map(|_| n))
            .with_context(|| format!("tensor `{name}` has an absurd shape {:?}", t.shape))?;
        let mut expect = Vec::with_capacity(t.shape.len());
        let mut acc = 1usize;
        for &d in t.shape.iter().rev() {
            expect.push(acc);
            acc = acc
                .checked_mul(d.max(1))
                .with_context(|| format!("tensor `{name}` has an absurd shape"))?;
        }
        expect.reverse();
        ensure!(
            t.stride == expect,
            "tensor `{name}` is not contiguous (stride {:?})",
            t.stride
        );
        let storage = entries
            .get(format!("{stem}data/{}", t.key).as_str())
            .with_context(|| format!("tensor `{name}`: storage `{}` missing", t.key))?;
        let start = t
            .offset
            .checked_mul(4)
            .with_context(|| format!("tensor `{name}` has an absurd offset"))?;
        let raw = storage
            .get(start..start.saturating_add(numel * 4))
            .with_context(|| format!("tensor `{name}` extends past its storage"))?;
        out.push((
            name,
            TorchTensorF32 {
                shape: t.shape,
                bytes: raw.to_vec(),
            },
        ));
    }
    Ok(out)
}
