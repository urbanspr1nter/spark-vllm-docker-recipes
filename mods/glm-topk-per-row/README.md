# glm-topk-per-row [SHELVED HARD 2026-07-20 — DO NOT retry on the 8-ring]

SECOND CRASH 2026-07-20: retry at gpu_memory_utilization 0.78 with a
fleet memory guard STILL froze nodes mid-ladder. Two lessons:
(1) 0.78 is not enough slack — the >400K indexer transient is larger
    than ~17GB or spikes faster than 5s sampling can see;
(2) the memory guard ran ON spark-01 — the node most likely to freeze —
    and died with it. Any future guard must run OFF-fleet (laptop,
    utility pair, anything not in the TP group).
Do not attempt >400K actual context on this cluster again without both
a much bigger headroom margin AND an off-fleet guard. 400K remains the
production ceiling. The CTA-ceiling fix itself remains valid and free
(parity: 25.14 tok/s, exact) — the memory envelope is the blocker.

TESTED on the 8-node ring, 2026-07-20:
- Parity at 400K: PASS — canary exact, decode 25.14 tok/s vs 24.0
  baseline (per-row kernel is FREE, possibly slightly faster).
- 450K boot: PASS — engine starts, KV sizes (727K tokens), canary exact.
  The topk.cu 48-CTA ceiling is definitively lifted.
- 430K-token actual-context needle: **CRASHED 3 NODES** (spark-01/05/07
  frozen, fleet power-cycle required). NVRM NV_ERR_NO_MEMORY on the head
  at prefill depth ~14 min in — DSA indexer transient allocations at
  >400K actual context overran unified memory (NOT a kernel bug; NOT
  heat; earlyoom can't catch driver-side exhaustion). Deepest context
  ever run before this was 319K — the 320-430K range was uncharted.

BEFORE RETRYING >400K actual context: lower gpu_memory_utilization
(0.85 -> ~0.78-0.80) to leave host slack for indexer transients, watch
`journalctl -k` for NV_ERR_NO_MEMORY during a staged depth ladder
(350K -> 380K -> 410K...), with a live free-memory monitor per node.
The mod itself is sound; the memory envelope is the open problem.

Lifts GLM-5.2's 400K max_model_len ceiling on DGX Spark (sm_121) by
routing the DSA indexer's decode top-k away from `persistent_topk`.

## Why 400K is the stock ceiling

`persistent_topk` launches cooperative CTA groups sized by context
length (~8,350 columns per CTA). GB10 fits 48 co-resident CTAs
(num_sms x occupancy); 400,000 tokens = exactly 48. 450,000 needs 54
and dies at kernel launch (`topk.cu:136`). The alternative FlashInfer
kernel (FilteredTopK) needs 128KB smem/block — sm_121 has ~101KB, so
it can never run there.

## What this mod does

One dispatch change in `vllm/model_executor/layers/sparse_attn_indexer.py`:
`use_persistent_topk = False`, so decode takes the existing else-branch,
`ops.top_k_per_row_decode` — vLLM's own per-row CUDA kernel with no
CTA/context coupling. The PREFILL path already uses the per-row family
(`top_k_per_row_prefill`) for everything, including every token of the
validated 400K deployment, so the kernel family is battle-tested on
this hardware. No weights, attention math, or selection semantics
change — top-k of the same logits, different kernel.

## Trade-off

persistent_topk exists because it is faster. The decode-speed cost of
per-row is UNMEASURED on GB10 — could be negligible, could be
noticeable. Measure before adopting.

## Test plan (needs the 8-node main ring; ~1 hour window)

1. Add this mod to glm-5.2-nvfp4.yaml `mods:`, keep max_model_len 400000.
   Launch. Gate: canaries exact (127*43=5461 etc.), decode tok/s vs the
   24.0 baseline — this measures the per-row speed tax at parity.
2. If exact + acceptable speed: bump max_model_len to 450000, relaunch.
   Gate: starts without topk.cu error (the old failure), canaries, then
   needle retrieval at 94% depth of a ~430K-token doc.
3. Push to 500000+ as desired; watch KV budget (fp8 KV at 500K on 8
   nodes is fine; the previous blocker was ONLY this kernel).
4. Keep/revert decision on speed numbers. Document results here.

## When to remove

If upstream adds a context-scalable topk for sm_12x or fixes
persistent_topk CTA scaling (watch flashinfer topk.cu and vLLM
sparse_attn_indexer.py dispatch).
