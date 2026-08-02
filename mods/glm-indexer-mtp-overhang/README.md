# glm-indexer-mtp-overhang

One-line fix for vLLM issue #46074: the DSA sparse indexer sizes
`expanded_block_table_buffer` from `max_model_len` alone, but MTP speculative
tokens can extend a request up to `num_speculative_tokens` past
`max_model_len`. When `max_model_len` is an exact multiple of `block_size`
there is no slack, and a decode step at ceiling-depth context needs one more
block than the buffer holds:

```
RuntimeError: The expanded size of the tensor (6250) must match the existing size (6251)
```

Our production config sits exactly on the trigger: `max_model_len=400000`,
`block_size=64` (400000/64 = 6250, remainder 0), MTP k=2.

## Fix

`+ 1` block of headroom on `max_num_blocks_per_req` in
`vllm/v1/attention/backends/mla/indexer.py`. Covers any
`num_speculative_tokens <= block_size`; costs a few KB.

## Provenance

- Upstream issue: vllm-project/vllm#46074 (jitrc, 32x H100, FLASHMLA_SPARSE —
  same root cause, still open as of 2026-07-26)
- Original patch: tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s
  `patches/fix-indexer-mtp-overhang.py` (4x Spark, vLLM ref ab66606) —
  re-anchored here for the dev1479 layout (buffer sizing moved to ~line 559,
  `get_total_cp_world_size()` renamed `get_kv_cache_shard_count()`).

## Verified against

vLLM 0.23.1rc1.dev1479 (eugr image 2026-07-25). Anchors on the
`get_kv_cache_shard_count()` cdiv block; fails loudly if the layout changes.

Drop this mod when #46074 is fixed upstream.
