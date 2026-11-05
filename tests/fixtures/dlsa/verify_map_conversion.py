#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dev-only cross-check of QW's Rust map conversion (`qw download --dlsa-map`). Not part of the build.

Reads the upstream torch `.pt` WITHOUT torch (stdlib zipfile + a restricted Unpickler) and compares every
tensor byte-for-byte against the converted safetensors file. Also prints both sha256 values.

    python3 verify_map_conversion.py <asset>.pt <converted>.safetensors
"""
import hashlib, json, pickle, struct, sys, zipfile


class Storage:
    def __init__(self, key): self.key = key


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


class U(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return lambda storage, off, size, stride, *rest: _Tensor(storage, off, size, stride)
        if (module, name) == ("collections", "OrderedDict"):
            return dict
        if (module, name) == ("torch", "FloatStorage"):
            return "FloatStorage"
        raise pickle.UnpicklingError(f"refusing {module}.{name}")

    def persistent_load(self, pid):
        tag, kind, key, _loc, _numel = pid
        assert tag == "storage" and kind == "FloatStorage", pid
        return Storage(key)


def main(pt, st):
    z = zipfile.ZipFile(pt)
    pkl = next(n for n in z.namelist() if n.endswith("data.pkl"))
    stem = pkl[: -len("data.pkl")]
    import io
    obj = U(io.BytesIO(z.read(pkl))).load()
    raw = open(st, "rb").read()
    hlen = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + hlen])
    body = raw[8 + hlen :]
    meta = json.loads(header.pop("__metadata__")["qw_dlsa_map"])
    assert meta["source_sha256"] == hashlib.sha256(open(pt, "rb").read()).hexdigest(), "source sha mismatch"
    assert len(header) == len(obj), (len(header), len(obj))
    for key, t in obj.items():
        layer, which = key.split(".")
        name = f"layers.{layer}.{'w_q_idx' if which == 'W_Q_idx' else 'w_k_idx'}"
        h = header[name]
        assert h["dtype"] == "F32" and tuple(h["shape"]) == t.size, (name, h, t.size)
        assert t.offset == 0 and t.stride == (t.size[1], 1)
        a, b = h["data_offsets"]
        assert body[a:b] == z.read(f"{stem}data/{t.storage.key}"), name
    print("OK", len(obj), "tensors identical")
    print("source sha256   ", hashlib.sha256(open(pt, "rb").read()).hexdigest())
    print("converted sha256", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:3])
