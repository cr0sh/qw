#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright Prefrontal-AI latent-sparse-attention contributors (upstream algorithm and reference code)
#
# Derived from https://github.com/Prefrontal-AI/latent-sparse-attention at commit
# 5ebea732d639576d2a7579b0de40e6252f9001f5 (Apache-2.0). Every algorithmic step below is a call into that
# commit's torch reference (src/lsa/core/{index_space,ward,reference,attention}.py); this file only generates
# synthetic inputs, calls the reference, checks invariants and writes the results.
"""OFFLINE DEV TOOL - NOT PART OF THE BUILD.

Generates the D-LSA (flat design) golden fixtures under tests/fixtures/dlsa/ from the upstream torch reference.
Nothing in cargo builds, tests or ships this script; Python and torch are not QW dependencies. The committed
`.safetensors` files and `manifest.json` are the canonical fixtures - Rust tests read those, never this script.

Environment (scratch only, never in the repo):

    git clone https://github.com/Prefrontal-AI/latent-sparse-attention "$SCRATCH/lsa"
    git -C "$SCRATCH/lsa" checkout 5ebea732d639576d2a7579b0de40e6252f9001f5
    uv venv --python 3.12 "$SCRATCH/venv"
    VIRTUAL_ENV="$SCRATCH/venv" uv pip install torch==2.13.0 pyyaml
    # (NumPy is not needed: the safetensors writer copies tensor memory with ctypes)
    curl -L -o "$SCRATCH/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt" \
      https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt

Regenerate (CPU only, deterministic - run it twice and compare `shasum -a 256 tests/fixtures/dlsa/*`):

    "$SCRATCH/venv/bin/python" tests/fixtures/dlsa/gen_fixtures.py \
      --lsa "$SCRATCH/lsa" --map "$SCRATCH/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"

The upstream checkout is imported from `<lsa>/src` (torch-only modules; no Triton/CUDA is touched). The script
refuses a checkout at another commit and a map whose sha256 differs from the release's.
"""

import argparse
import ctypes
import hashlib
import json
import platform
import struct
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "5ebea732d639576d2a7579b0de40e6252f9001f5"
MAP_NAME = "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"
MAP_SHA256 = "24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b"
MAP_LAYER = 3  # first full-attention decoder layer (0-based), the map's first entry

# safetensors dtype names (https://github.com/huggingface/safetensors, format spec) for the torch dtypes written
SAFETENSORS_DTYPES = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ward_labels_kernel_policy(pool_keys, pool_sizes, bucket_count, bucket_cap, max_rounds=256):
    """Ward rounds with the served Triton kernel's tie policy, in torch - the policy QW follows.

    The merge cost, cap and mutual rule are reference ward.py:60-90 line for line (fp32 Gram). Ties are broken
    as core/kernels/triton/ward.py:154-204 breaks them: a bucket's partner is the lowest index among equal-cost
    candidates (packed (cost, column) key, min), and when more mutual pairs exist than merges are left, pairs are
    ranked by the lower bucket's packed (cost, partner) key - an equal-cost tie goes to the pair whose higher
    (absorbed) bucket has the lower index. The reference instead ranks with an unstable argsort (ward.py:76), so
    its choice among such ties is implementation-defined.
    """
    blocks, pools, _ = pool_keys.shape
    key_sums, sizes = pool_keys.clone(), pool_sizes.clone()
    alive = torch.ones(blocks, pools, dtype=torch.bool)
    parent = torch.arange(pools).expand(blocks, pools).clone()
    index = torch.arange(pools).expand(blocks, pools)
    diagonal = torch.eye(pools, dtype=torch.bool)
    for _ in range(max_rounds):
        merges_left = alive.sum(1) - bucket_count
        if not (merges_left > 0).any():
            break
        gram = torch.bmm(key_sums, key_sums.transpose(1, 2))
        squares = gram.diagonal(dim1=1, dim2=2)
        merged = (squares[:, :, None] + squares[:, None, :] + 2 * gram).clamp_min(0).sqrt()
        alignments = squares.clamp_min(0).sqrt()
        costs = alignments[:, :, None] + alignments[:, None, :] - merged
        allowed = alive[:, :, None] & alive[:, None, :] & ~diagonal
        allowed &= sizes[:, :, None] + sizes[:, None, :] <= bucket_cap
        costs = costs.masked_fill(~allowed, float("inf"))
        best_cost = costs.min(2).values
        tied = (costs == best_cost[:, :, None]) & allowed
        best = torch.where(tied, index[:, None, :], pools).min(2).values.clamp_max(pools - 1)
        mutual = (best_cost < float("inf")) & (best.gather(1, best) == index) & (index < best)
        cost_key = best_cost.masked_fill(~mutual, float("inf"))
        partner_key = best.masked_fill(~mutual, pools)
        by_partner = torch.argsort(partner_key, dim=1, stable=True)
        order = by_partner.gather(1, torch.argsort(cost_key.gather(1, by_partner), dim=1, stable=True))
        rank = torch.empty_like(order).scatter_(1, order, index.clone())
        mutual &= rank < merges_left.clamp_min(0)[:, None]
        if not mutual.any():
            break
        block_ids, kept = mutual.nonzero(as_tuple=True)
        absorbed = best[block_ids, kept]
        key_sums[block_ids, kept] += key_sums[block_ids, absorbed]
        sizes[block_ids, kept] += sizes[block_ids, absorbed]
        alive[block_ids, absorbed] = False
        parent[block_ids, absorbed] = kept
    for _ in range(pools.bit_length() + 1):
        parent = parent.gather(1, parent)
    return (torch.cumsum(alive.long(), 1) - 1).gather(1, parent)


def write_safetensors(path: Path, tensors: dict, metadata: dict) -> None:
    """Write a safetensors file by hand (8-byte LE header length, JSON header, packed LE data), no extra package."""
    header = {"__metadata__": {key: str(value) for key, value in metadata.items()}}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        tensor = tensor.detach().contiguous().cpu()
        # Raw bytes straight from the CPU tensor's memory (no NumPy); a little-endian host is asserted in main()
        data = ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()) if tensor.numel() else b""
        header[name] = {
            "dtype": SAFETENSORS_DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for data in blobs:
            handle.write(data)


# ----------------------------------------------------------------------------------------------------------------
# Synthetic inputs
# ----------------------------------------------------------------------------------------------------------------


def fp16_exact(tensor):
    """Round to fp16 and back, so the fp16 the fixture stores is exactly what the reference computed with."""
    return tensor.half().float()


def synthetic_heads(gen, key_count, query_heads, kv_heads, head_dim, topics, run_mean, window, repeat_period=0):
    """Pre-RoPE, post-QK-norm query/key/value heads with topical structure, so selection is not uniform noise.

    Keys come in runs of a topic; a query at position p is drawn towards the topic of one earlier key outside its
    window (a "needle"), so the index has something to find. Everything is fp16-representable.
    """
    centers = torch.randn(topics, kv_heads, head_dim, generator=gen)
    labels = []
    while len(labels) < key_count:
        topic = int(torch.randint(topics, (1,), generator=gen))
        length = 1 + int(torch.randint(2 * run_mean, (1,), generator=gen))
        labels.extend([topic] * length)
    labels = torch.tensor(labels[:key_count])
    noise = torch.randn(key_count, kv_heads, head_dim, generator=gen)
    if repeat_period:
        # A repeated text: every key equal to the one `repeat_period` before it - ties everywhere downstream
        noise = noise[:repeat_period].repeat(key_count // repeat_period + 1, 1, 1)[:key_count]
        labels = labels[:repeat_period].repeat(key_count // repeat_period + 1)[:key_count]
    keys = fp16_exact(0.7 * centers[labels] + 0.7 * noise)
    values = fp16_exact(torch.randn(key_count, kv_heads, head_dim, generator=gen))
    group = query_heads // kv_heads
    queries = torch.empty(key_count, query_heads, head_dim)
    for position in range(key_count):
        far = position - window
        source = int(torch.randint(far, (1,), generator=gen)) if far > 0 else position
        target = centers[labels[source]]  # [kv_heads, head_dim]
        queries[position] = target.repeat_interleave(group, dim=0) + 0.8 * torch.randn(
            query_heads, head_dim, generator=gen
        )
    return fp16_exact(queries), keys, values


# ----------------------------------------------------------------------------------------------------------------
# The pipeline, every step a call into the upstream reference
# ----------------------------------------------------------------------------------------------------------------


def index_vectors(q, k, rotary_dim, query_map, key_map):
    """Index queries and index keys exactly as distillation's index space computes them.

    position_free (index_space.py:62) drops the rotated prefix, regroup (index_space.py:110) sums each KV group's
    query heads and concatenates KV heads then unit-normalizes, project (index_space.py:405) applies the map and
    unit-normalizes again (the first normalization is a positive scalar and cancels).

    Args: q [T, Hq, dh], k [T, Hkv, dh]; maps [Hkv*(dh-rot), d].
    Returns: index_queries [T, d] fp32, index_keys [T, d] fp32, regrouped (un-projected) query/key rows.
    """
    layout = RotaryLayout(Rotary.PREFIX, rotary_dim) if rotary_dim else RotaryLayout(Rotary.NONE, 0)
    free_q, free_k = position_free(q.transpose(0, 1)[None], k.transpose(0, 1)[None], layout)
    grouped_q, concat_k = regroup_partial(free_q, free_k)
    regrouped_q, regrouped_k = regroup(free_q, free_k)
    index_queries = project(regrouped_q[0], query_map)
    index_keys = project(regrouped_k[0], key_map)
    return index_queries, index_keys, grouped_q[0], concat_k[0]


def picks_and_scores(index_queries, positions, buckets, top_p):
    """The bucket scores and the top-p picks of each query, the same expressions as reference.read/_read_row.

    reference.py:228 scores every bucket; reference.py:262-264 keeps the buckets of blocks closed before the
    query's own and argsorts them best first, a tie to the lower bucket (stable sort).
    """
    scores = index_queries.float() @ buckets.bucket_centroids.T
    picked = torch.full((len(positions), top_p), -1, dtype=torch.int32)
    visible_counts = torch.zeros(len(positions), dtype=torch.int32)
    margins = []
    for row, position in enumerate(positions.tolist()):
        visible = min(position // buckets.cluster_block_size * buckets.block_bucket_count, scores.shape[1])
        order = torch.argsort(scores[row, :visible], descending=True, stable=True)
        chosen = order[:top_p]
        picked[row, : len(chosen)] = chosen.to(torch.int32)
        visible_counts[row] = visible
        if visible > top_p:
            ranked = scores[row, order]
            margins.append(float(ranked[top_p - 1] - ranked[top_p]))
    return scores, picked, visible_counts, margins


def dense_attention(q, k, v, position, scale):
    """Plain causal GQA softmax attention of one query row over keys [0, position], fp32 - an independent oracle."""
    group = q.shape[0] // k.shape[1]
    keys = k[: position + 1].float().repeat_interleave(group, dim=1)  # [S, Hq, dh]
    values = v[: position + 1].float().repeat_interleave(group, dim=1)
    logits = torch.einsum("hd,shd->hs", q.float(), keys) * scale
    return torch.einsum("hs,shd->hd", torch.softmax(logits, dim=-1), values)


def check_list(selected, length, position, config, all_keys_expected):
    """Every list holds its window and its sinks, ids below the position, no duplicates; dense when expected."""
    ids = selected[:length].tolist()
    assert all(i >= 0 for i in ids) and all(i < 0 for i in selected[length:].tolist())
    assert len(set(ids)) == len(ids), "an id is read twice"
    assert max(ids) <= position
    window_start = max(0, position + 1 - config["window_size"])
    required = set(range(window_start, position + 1)) | set(range(min(config["sink_size"], position + 1)))
    assert required <= set(ids), "a list is missing its window or sink"
    if all_keys_expected:
        assert set(ids) == set(range(position + 1)), "a short context must read every causal key"


def run_case(name, config, q, k, v, maps, prefill_len, attention_rows, rotary_dim, scale, store_qkv_rows_only):
    """Run the flat design over one case and return its tensors and its manifest entry.

    maps: {label: (query_map, key_map)} - one set of outputs per map, prefixed `<label>.` when there are several.
    """
    key_count = k.shape[0]
    window, sink = config["window_size"], config["sink_size"]
    top_k, top_p = config["top_k"], config["top_p"]
    pool, bucket, cap = config["pool_size"], config["bucket_size"], config["bucket_cap"]
    tensors = {}
    if store_qkv_rows_only:
        tensors["q_rows"] = q[attention_rows].half()
    else:
        tensors["q"] = q.half()
    tensors["k"] = k.half()
    tensors["v"] = v.half()
    tensors["attention_rows"] = attention_rows.to(torch.int32)
    diagnostics = {}
    positions = torch.arange(key_count)
    for label, (query_map, key_map) in maps.items():
        prefix = f"{label}." if len(maps) > 1 else ""
        index_queries, index_keys, grouped_q, concat_k = index_vectors(q, k, rotary_dim, query_map, key_map)
        # RoPE invariance: rotating the prefix dims (any position-dependent map of them) leaves the index unchanged
        if rotary_dim:
            twisted_q, twisted_k = q.clone(), k.clone()
            twisted_q[..., :rotary_dim] = twisted_q[..., :rotary_dim].flip(-1) * 3.0
            twisted_k[..., :rotary_dim] = -twisted_k[..., :rotary_dim]
            iq2, ik2, _, _ = index_vectors(twisted_q, twisted_k, rotary_dim, query_map, key_map)
            assert torch.equal(iq2, index_queries) and torch.equal(ik2, index_keys)
        # The cache after the prefill, and the cache as it stands at each decode step
        prefill_buckets = build_buckets(index_keys[:prefill_len], pool, bucket, cap, window)
        final_buckets = build_buckets(index_keys, pool, bucket, cap, window)
        blocks = key_count // window
        pools = pool_keys(index_keys[: blocks * window].view(blocks, window, -1), pool)
        labels = ward_merge(pools, torch.full(pools.shape[:2], float(pool)), window // bucket, cap)
        # No oversubscribed equal-cost tie here: the kernel's tie policy gives the reference's labels
        assert torch.equal(ward_labels_kernel_policy(pools, torch.full(pools.shape[:2], float(pool)), window // bucket, cap), labels)
        # Each closed block is clustered once and never again: the prefill's buckets are a prefix of the final ones
        rows = prefill_buckets.bucket_centroids.shape[0]
        assert torch.equal(prefill_buckets.bucket_centroids, final_buckets.bucket_centroids[:rows])
        assert torch.equal(prefill_buckets.bucket_members, final_buckets.bucket_members[:rows])
        selected = torch.full((key_count, top_k + window + sink), -1, dtype=torch.int32)
        lengths = torch.zeros(key_count, dtype=torch.int32)
        sel, ln = read(index_queries[:prefill_len], positions[:prefill_len], prefill_buckets, top_p, top_k, sink, window)
        selected[:prefill_len], lengths[:prefill_len] = sel, ln
        for position in range(prefill_len, key_count):
            # Incremental decode: the cache holds keys [0, position], the new key included
            step_buckets = build_buckets(index_keys[: position + 1], pool, bucket, cap, window)
            sel, ln = read(index_queries[position : position + 1], positions[position : position + 1],
                           step_buckets, top_p, top_k, sink, window)
            selected[position], lengths[position] = sel[0], ln[0]
        # A query's read depends on its position alone, not on when the cache was built
        sel_all, ln_all = read(index_queries, positions, final_buckets, top_p, top_k, sink, window)
        assert torch.equal(sel_all, selected) and torch.equal(ln_all, lengths)
        scores, picked, visible_counts, margins = picks_and_scores(index_queries, positions, final_buckets, top_p)
        for position in range(key_count):
            # The picks expanded by reference.expand_buckets (reference.py:268) are the list read() returned
            chosen = picked[position][picked[position] >= 0].long()
            expanded = expand_buckets(chosen, position, final_buckets, top_k, sink, window)
            assert torch.equal(expanded, selected[position, : lengths[position]])
            dense_expected = position + 1 <= sink + top_k + window and int(visible_counts[position]) <= top_p
            check_list(selected[position], int(lengths[position]), position, config, dense_expected)
        # Attention: the sparse read (attention.py:65) and a dense oracle for the same rows
        outputs, dense_outputs = [], []
        for position in attention_rows.tolist():
            out = topk_attention(q[None, position : position + 1], k[None, : position + 1], v[None, : position + 1],
                                 selected[None, position : position + 1], window, sink, scale, query_start=position)
            outputs.append(out[0, 0])
            dense_outputs.append(dense_attention(q[position], k, v, position, scale))
        outputs, dense_outputs = torch.stack(outputs), torch.stack(dense_outputs)
        complete = [int(lengths[p]) == p + 1 for p in attention_rows.tolist()]
        for row, full in enumerate(complete):
            if full:
                assert torch.allclose(outputs[row], dense_outputs[row], atol=1e-5, rtol=1e-5)
        sparse_rows = [row for row, full in enumerate(complete) if not full]
        # A map that is fp16-exact (the random ones) is stored as fp16, losslessly; the real map stays fp32
        for map_name, matrix in (("query_map", query_map), ("key_map", key_map)):
            exact = torch.equal(matrix.half().float(), matrix)
            tensors[f"{prefix}{map_name}"] = matrix.half() if exact else matrix.clone()
        tensors[f"{prefix}grouped_queries"] = grouped_q[attention_rows].clone()
        tensors[f"{prefix}concat_keys"] = concat_k.clone() if key_count * concat_k.shape[1] <= 65536 else None
        tensors[f"{prefix}index_keys"] = index_keys
        tensors[f"{prefix}index_queries"] = index_queries
        tensors[f"{prefix}pool_keys"] = pools
        tensors[f"{prefix}ward_labels"] = labels.to(torch.int32)
        tensors[f"{prefix}bucket_centroids"] = final_buckets.bucket_centroids
        tensors[f"{prefix}bucket_members"] = final_buckets.bucket_members
        tensors[f"{prefix}bucket_sizes"] = final_buckets.bucket_sizes
        tensors[f"{prefix}visible_buckets"] = visible_counts
        tensors[f"{prefix}bucket_scores"] = scores
        tensors[f"{prefix}picked_buckets"] = picked
        tensors[f"{prefix}selected"] = selected
        tensors[f"{prefix}selected_lengths"] = lengths
        tensors[f"{prefix}attention_out"] = outputs
        tensors[f"{prefix}dense_out"] = dense_outputs if not store_qkv_rows_only else None
        relative = [
            float((outputs[row] - dense_outputs[row]).norm() / dense_outputs[row].norm()) for row in sparse_rows
        ]
        diagnostics[label] = {
            "prefill_buckets": int(prefill_buckets.bucket_centroids.shape[0]),
            "final_buckets": int(final_buckets.bucket_centroids.shape[0]),
            "rows_reading_every_key": int(sum(int(lengths[p]) == p + 1 for p in range(key_count))),
            "rows_truncated_by_top_p": int(sum(1 for c in visible_counts.tolist() if c > top_p)),
            "min_top_p_boundary_gap": min(margins) if margins else None,
            "sparse_vs_dense_relative_error_max": max(relative) if relative else None,
            "sparse_vs_dense_relative_error_mean": sum(relative) / len(relative) if relative else None,
        }
    tensors = {name: tensor for name, tensor in tensors.items() if tensor is not None}
    return tensors, diagnostics


def run_index_case(config, index_keys, index_queries, positions):
    """Index-only case at the served Qwen defaults: bf16 index keys (as the serving cache clusters them) in,
    buckets and lists out. No attention - the budget needs a context too long to store K/V for."""
    window, sink = config["window_size"], config["sink_size"]
    top_k, top_p = config["top_k"], config["top_p"]
    pool, bucket, cap = config["pool_size"], config["bucket_size"], config["bucket_cap"]
    buckets = build_buckets(index_keys, pool, bucket, cap, window)
    blocks = index_keys.shape[0] // window
    pools = pool_keys(index_keys[: blocks * window].float().view(blocks, window, -1), pool)
    labels = ward_merge(pools, torch.full(pools.shape[:2], float(pool)), window // bucket, cap)
    assert torch.equal(ward_labels_kernel_policy(pools, torch.full(pools.shape[:2], float(pool)), window // bucket, cap), labels)
    selected, lengths = read(index_queries, positions, buckets, top_p, top_k, sink, window)
    # The same rows read one at a time against the cache as it stood at each one (incremental decode)
    for row, position in enumerate(positions.tolist()):
        if position + 1 < window:
            # reference.build_buckets raises on a cache with no closed block (ward.py:115 reduces an empty
            # tensor); such a query sees no bucket either way, so the final cache stands in for it
            continue
        step = build_buckets(index_keys[: position + 1], pool, bucket, cap, window)
        one, count = read(index_queries[row : row + 1], positions[row : row + 1], step, top_p, top_k, sink, window)
        assert torch.equal(one[0], selected[row]) and int(count[0]) == int(lengths[row])
    scores, picked, visible_counts, margins = picks_and_scores(index_queries, positions, buckets, top_p)
    for row, position in enumerate(positions.tolist()):
        dense_expected = position + 1 <= sink + top_k + window and int(visible_counts[row]) <= top_p
        check_list(selected[row], int(lengths[row]), position, config, dense_expected)
    tensors = {
        "index_keys": index_keys,
        "index_queries": index_queries,
        "positions": positions.to(torch.int32),
        "ward_labels": labels.to(torch.int32),
        "bucket_centroids": buckets.bucket_centroids,
        "bucket_members": buckets.bucket_members,
        "bucket_sizes": buckets.bucket_sizes,
        "visible_buckets": visible_counts,
        "bucket_scores": scores,
        "picked_buckets": picked,
        "selected": selected,
        "selected_lengths": lengths,
    }
    diagnostics = {
        "buckets": int(buckets.bucket_centroids.shape[0]),
        "rows_reading_every_key": int(sum(int(lengths[r]) == p + 1 for r, p in enumerate(positions.tolist()))),
        "rows_truncated_by_top_p": int(sum(1 for c in visible_counts.tolist() if c > top_p)),
        "min_top_p_boundary_gap": min(margins) if margins else None,
    }
    return tensors, diagnostics


def serving_numerics_drift(q, k, rotary_dim, query_map, key_map, config, positions):
    """How many lists change under the served numerics (layer_index.py:296-355, clustering.py:117-130,
    selector.py:403-409) - bf16 maps and products, bf16 index keys and centroids, bf16 queries scored in fp32 -
    against the fp32 reference. A diagnostic only; the fixtures are the fp32 reference."""
    _, _, grouped_q, concat_k = index_vectors(q, k, rotary_dim, query_map, key_map)
    keys16 = F.normalize((concat_k.to(torch.bfloat16) @ key_map.to(torch.bfloat16)).float(), dim=-1).to(torch.bfloat16)
    queries = F.normalize((grouped_q.to(torch.bfloat16) @ query_map.to(torch.bfloat16)).float(), dim=-1)
    queries = queries.to(torch.bfloat16).float()
    window, sink = config["window_size"], config["sink_size"]
    top_k, top_p = config["top_k"], config["top_p"]
    buckets = build_buckets(keys16, config["pool_size"], config["bucket_size"], config["bucket_cap"], window)
    buckets = buckets._replace(bucket_centroids=buckets.bucket_centroids.to(torch.bfloat16).float())
    served, served_lengths = read(F.normalize(queries, dim=-1), positions, buckets, top_p, top_k, sink, window)
    _, index_keys, _, _ = index_vectors(q, k, rotary_dim, query_map, key_map)
    index_queries = index_vectors(q, k, rotary_dim, query_map, key_map)[0]
    ref_buckets = build_buckets(index_keys, config["pool_size"], config["bucket_size"], config["bucket_cap"], window)
    reference, reference_lengths = read(index_queries, positions, ref_buckets, top_p, top_k, sink, window)
    changed, overlaps = 0, []
    for row in range(len(positions)):
        a = set(served[row, : served_lengths[row]].tolist())
        b = set(reference[row, : reference_lengths[row]].tolist())
        if a != b:
            changed += 1
        overlaps.append(len(a & b) / len(a | b))
    label_changes = int((buckets.bucket_members != ref_buckets.bucket_members).any(1).sum())
    return {
        "rows": len(positions),
        "rows_with_a_different_list": changed,
        "min_jaccard": min(overlaps),
        "mean_jaccard": sum(overlaps) / len(overlaps),
        "buckets_with_different_members": label_changes,
        "total_buckets": int(buckets.bucket_members.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lsa", type=Path, required=True, help="upstream checkout at the pinned commit")
    parser.add_argument("--map", type=Path, required=True, help=f"path to {MAP_NAME}")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    commit = subprocess.run(["git", "-C", str(args.lsa), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"{args.lsa} is at {commit}, the fixtures are pinned to {PINNED_COMMIT}")
    map_sha = sha256_file(args.map)
    if map_sha != MAP_SHA256:
        raise SystemExit(f"{args.map} has sha256 {map_sha}, the release's is {MAP_SHA256}")
    sys.path.insert(0, str(args.lsa / "src"))

    global torch, F, RotaryLayout, Rotary, position_free, regroup, regroup_partial, project
    global pool_keys, ward_merge, build_buckets, read, expand_buckets, topk_attention
    import torch
    import torch.nn.functional as F
    from lsa.constants import Rotary
    from lsa.core.attention import topk_attention
    from lsa.core.index_space import RotaryLayout, position_free, project, regroup, regroup_partial
    from lsa.core.reference import build_buckets, expand_buckets, read
    from lsa.core.ward import pool_keys, ward_merge
    from lsa.sglang.settings.design import IndexerConfig

    SAFETENSORS_DTYPES.update({torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
                               torch.int32: "I32", torch.int64: "I64"})
    assert sys.byteorder == "little", "safetensors is little-endian and the writer copies raw memory"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    served = IndexerConfig()  # the paper's flat design - every Qwen eval config leaves model.indexer at these
    served_config = {
        "window_size": served.window_size, "sink_size": served.sink_size, "top_k": served.top_k,
        "top_p": served.top_p, "index_dim": served.index_dim, "pool_size": served.pool_size,
        "bucket_size": served.bucket_size, "bucket_cap": served.bucket_cap, "top_level": served.top_level,
    }
    assert served_config == {"window_size": 512, "sink_size": 4, "top_k": 4096, "top_p": 1024, "index_dim": 64,
                             "pool_size": 4, "bucket_size": 8, "bucket_cap": 16, "top_level": 0}
    assert served.early_budget is None and served.bucket_shape is None

    real = torch.load(args.map, map_location="cpu", weights_only=True)
    real_query_map = real[f"{MAP_LAYER}.W_Q_idx"].float()
    real_key_map = real[f"{MAP_LAYER}.W_K_idx"].float()
    assert real_query_map.shape == (768, 64) and real_key_map.shape == (768, 64)

    def validated(config):
        """Refuse a fixture config the served design's own validation would refuse (design.py:213-285)."""
        IndexerConfig(**{key: config[key] for key in ("window_size", "sink_size", "top_k", "top_p", "index_dim",
                                                      "pool_size", "bucket_size", "bucket_cap")})
        return config

    manifest = {
        "about": "D-LSA flat-design golden fixtures; see README.md. Generated by gen_fixtures.py (offline tool).",
        "upstream": "https://github.com/Prefrontal-AI/latent-sparse-attention",
        "upstream_commit": PINNED_COMMIT,
        "license": "Apache-2.0 (derived from the upstream reference)",
        "map": {"asset": MAP_NAME, "sha256": MAP_SHA256, "layer": MAP_LAYER,
                "url": f"https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/{MAP_NAME}"},
        "served_defaults": served_config,
        "generator": {"python": platform.python_version(), "torch": torch.__version__},
        "cases": {},
    }

    def emit(name, config, tensors, diagnostics, shapes):
        path = args.out / f"{name}.safetensors"
        metadata = {"case": name, "upstream_commit": PINNED_COMMIT, "config": json.dumps(config, sort_keys=True)}
        write_safetensors(path, tensors, metadata)
        manifest["cases"][name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "config": config,
            "shapes": shapes,
            "diagnostics": diagnostics,
            "tensors": {key: {"dtype": SAFETENSORS_DTYPES[value.dtype], "shape": list(value.shape)}
                        for key, value in tensors.items()},
        }

    # --- tiny: every stage at toy sizes, Q/K/V whole, sparse enough to cut buckets --------------------------------
    tiny = validated({"window_size": 16, "sink_size": 2, "top_k": 12, "top_p": 4, "index_dim": 8, "pool_size": 2,
                      "bucket_size": 4, "bucket_cap": 8, "prefill_len": 78, "key_count": 84})
    shapes = {"query_heads": 6, "kv_heads": 2, "head_dim": 16, "rotary_dim": 4}
    gen = torch.Generator().manual_seed(1)
    q, k, v = synthetic_heads(gen, 84, 6, 2, 16, topics=6, run_mean=3, window=16)
    key_dim = 2 * (16 - 4)
    maps = {"random": (fp16_exact(torch.randn(key_dim, 8, generator=gen)), fp16_exact(torch.randn(key_dim, 8, generator=gen)))}
    tensors, diagnostics = run_case("tiny", tiny, q, k, v, maps, 78, torch.arange(84), 4, 16**-0.5, False)
    emit("tiny", tiny, tensors, diagnostics, shapes)

    # --- tiny_ties: a repeated text (period 8 = two pools) - equal pools, equal merge costs, equal bucket scores --
    ties = validated(dict(tiny))
    gen = torch.Generator().manual_seed(2)
    q, k, v = synthetic_heads(gen, 84, 6, 2, 16, topics=6, run_mean=3, window=16, repeat_period=8)
    maps = {"random": (fp16_exact(torch.randn(key_dim, 8, generator=gen)), fp16_exact(torch.randn(key_dim, 8, generator=gen)))}
    tensors, diagnostics = run_case("tiny_ties", ties, q, k, v, maps, 78, torch.arange(84), 4, 16**-0.5, False)
    emit("tiny_ties", ties, tensors, diagnostics, shapes)

    # --- short_dense: a context no longer than sink + top_k + window - D-LSA must equal dense attention ---------
    short = validated({"window_size": 16, "sink_size": 2, "top_k": 24, "top_p": 8, "index_dim": 8, "pool_size": 2,
                       "bucket_size": 4, "bucket_cap": 8, "prefill_len": 36, "key_count": 42})
    gen = torch.Generator().manual_seed(3)
    q, k, v = synthetic_heads(gen, 42, 6, 2, 16, topics=6, run_mean=3, window=16)
    maps = {"random": (fp16_exact(torch.randn(key_dim, 8, generator=gen)), fp16_exact(torch.randn(key_dim, 8, generator=gen)))}
    tensors, diagnostics = run_case("short_dense", short, q, k, v, maps, 36, torch.arange(42), 4, 16**-0.5, False)
    assert torch.equal(tensors["selected_lengths"], torch.arange(1, 43, dtype=torch.int32))
    assert torch.allclose(tensors["attention_out"], tensors["dense_out"], atol=1e-5, rtol=1e-5)
    emit("short_dense", short, tensors, diagnostics, shapes)

    # --- ward_tie_policy: equal-cost mutual pairs oversubscribing the merges left - where tie policies differ ---
    # Pools e1, e2, e2, e1 of one key each, 3 buckets, cap 4: pairs (0,3) and (1,2) both cost 0 and are mutual,
    # one merge is left. The served kernel's packed (cost, partner) rank merges (1,2); the reference's unstable
    # argsort picks either. QW follows the kernel.
    tie_pools = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])
    tie_sizes = torch.ones(1, 4)
    kernel_labels = ward_labels_kernel_policy(tie_pools, tie_sizes, 3, 4)
    assert kernel_labels.tolist() == [[0, 1, 1, 2]], kernel_labels
    reference_labels = ward_merge(tie_pools, tie_sizes, 3, 4)
    tie_config = {"bucket_count": 3, "bucket_cap": 4, "pool_size": 1}
    tie_tensors = {"pool_keys": tie_pools, "pool_sizes": tie_sizes, "ward_labels": kernel_labels.to(torch.int32),
                   "reference_ward_labels": reference_labels.to(torch.int32)}
    emit("ward_tie_policy", tie_config, tie_tensors,
         {"reference_labels_this_torch": reference_labels.tolist(), "kernel_policy_labels": kernel_labels.tolist(),
          "reference_is_implementation_defined": True}, {"index_dim": 2})

    # --- qwen_layer: one Qwen3.8-27B full-attention layer's shapes, a scaled-down budget, random and real maps ---
    layer = validated({"window_size": 64, "sink_size": 4, "top_k": 64, "top_p": 16, "index_dim": 64,
                       "pool_size": 4, "bucket_size": 8, "bucket_cap": 16, "prefill_len": 252, "key_count": 260})
    qwen_shapes = {"query_heads": 24, "kv_heads": 4, "head_dim": 256, "rotary_dim": 64}
    gen = torch.Generator().manual_seed(4)
    q, k, v = synthetic_heads(gen, 260, 24, 4, 256, topics=24, run_mean=10, window=64)
    maps = {
        "random": (fp16_exact(torch.randn(768, 64, generator=gen) / 768**0.5),
                   fp16_exact(torch.randn(768, 64, generator=gen) / 768**0.5)),
        "real": (real_query_map, real_key_map),
    }
    # Prefill rows below, at and past the window; decode rows 252..259 cross the block boundary at 256
    rows = torch.tensor([63, 64, 191, 251, 252, 255, 256, 257, 258, 259])
    tensors, diagnostics = run_case("qwen_layer", layer, q, k, v, maps, 252, rows, 64, 256**-0.5, True)
    diagnostics["serving_numerics_drift_real_map"] = serving_numerics_drift(
        q, k, 64, real_query_map, real_key_map, layer, torch.arange(260))
    emit("qwen_layer", layer, tensors, diagnostics, qwen_shapes)

    # --- qwen_index_defaults: the served defaults (window 512, top-k 4096, top-p 1024) at 9,216 keys -------------
    defaults = dict(served_config)
    defaults.pop("top_level")
    defaults["key_count"] = 9216
    gen = torch.Generator().manual_seed(5)
    # Synthetic position-free key/query input [T, 768] (4 KV heads x 192 dims), mapped by the real layer-3 map
    centers = torch.randn(48, 768, generator=gen)
    labels, topic = [], 0
    while len(labels) < 9216:
        topic = int(torch.randint(48, (1,), generator=gen))
        labels.extend([topic] * (1 + int(torch.randint(40, (1,), generator=gen))))
    labels = torch.tensor(labels[:9216])
    concat = 0.7 * centers[labels] + 0.7 * torch.randn(9216, 768, generator=gen)
    index_keys = F.normalize(concat @ real_key_map, dim=-1).to(torch.bfloat16)  # as the serving cache holds them
    positions = torch.tensor([100, 511, 512, 1023, 1024, 4611, 4612, 6000, 8191, 8703, 8704, 9000, 9214, 9215])
    grouped = 6 * centers[labels[(positions - 600).clamp_min(0)]] + 2.0 * torch.randn(len(positions), 768, generator=gen)
    index_queries = F.normalize(grouped @ real_query_map, dim=-1)
    tensors, diagnostics = run_index_case(defaults, index_keys, index_queries, positions)
    emit("qwen_index_defaults", defaults, tensors, diagnostics, {"key_dim": 768, "index_dim": 64})

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    total = sum(case["bytes"] for case in manifest["cases"].values())
    print(json.dumps({name: case["diagnostics"] for name, case in manifest["cases"].items()}, indent=1))
    print(f"wrote {len(manifest['cases'])} cases, {total} bytes")


if __name__ == "__main__":
    main()
