# minimax-m3-swigluoai

Fixes MiniMax-M3-NVFP4 producing fluent garbage output on the MARLIN MoE
backend (the auto-picked NVFP4 backend on DGX Spark / sm_121).

## The problem

M3's FFN activation is SwiGLU-OAI ("swigluoai_uninterleave") with
per-model alpha/beta constants. The model sets `swiglu_alpha` /
`swiglu_beta` on the FusedMoE layer, and the Marlin fused-MoE kernel
fully supports them — but the NVFP4 quant-config plumbing between the
two drops them at three joints:

- `quantization/modelopt.py`: the `make_nvfp4_moe_quant_config(...)`
  call passes `swiglu_limit` but not alpha/beta;
- `fused_moe/oracle/nvfp4.py`: `make_nvfp4_moe_quant_config` doesn't
  accept them, and its MARLIN branch doesn't forward them;
- `fused_moe/config.py`: `nvfp4_w4a16_moe_quant_config` doesn't accept
  or forward them into `FusedMoEQuantConfig.make` (whose
  `gemm1_alpha`/`gemm1_beta` fields already exist).

`MarlinExperts.__init__` then defaults to `alpha=1.0, beta=0.0` — the
wrong activation on every expert of every layer. The model loads and
serves, but output is fluent token soup. A 5-token greedy
`/v1/completions` probe is enough to reproduce (sparse attention is
barely engaged at that length, which is how the MoE layer was isolated
as the culprit).

The other backends can't dodge this: FLASHINFER_CUTLASS refuses
SWIGLUOAI_UNINTERLEAVE at startup, FLASHINFER_B12X hits a CUDA illegal
memory access at warmup on sm_121, FLASHINFER_TRTLLM is not eligible.
MARLIN is the only runnable backend, so this fix is required.

Root cause was identified independently by baristankut (NVIDIA forum
thread 376979, working 4x DGX Spark deployment, ~31 tok/s, 1M context)
and confirmed by our black-box bisection. Unreported upstream as of
2026-07-18; adjacent upstream work: vLLM issue #45859, PR #43589.

## The fix

`run.sh` threads `swiglu_alpha`/`swiglu_beta` from the layer through the
quant-config builders into the quant config, mirroring the exact pattern
`quark/quark_moe.py` already uses for its MoE path. Five exact-string
edits across the three files above; idempotent; refuses loudly if
upstream code has drifted. No kernel changes — the receiving side
(`FusedMoEQuantConfig` fields, `MarlinExperts` plumbing, the fused
activation kernel) already exists upstream.

## Verification

First light on 4x DGX Spark pending (TP=4 ring). Gate sequence: raw
`/v1/completions` greedy probe (garbage reproduces instantly when
broken), then canaries, then vision probe, then long-context.

## When to remove

When upstream threads these params through the NVFP4 path (watch vLLM
issue #45859 and anything touching `nvfp4_w4a16_moe_quant_config`).
The patches self-detect an already-fixed upstream and skip.
