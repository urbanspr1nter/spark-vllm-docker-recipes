#!/bin/bash
set -euo pipefail

# MiniMax-M3-NVFP4 on the MARLIN MoE backend: fix the dropped SwiGLU-OAI
# activation parameters (gemm1_alpha / gemm1_beta).
#
# M3's FFN uses the swigluoai_uninterleave activation with per-model
# alpha/beta constants. The model sets swiglu_alpha/swiglu_beta on the
# FusedMoE layer, MarlinExperts reads them from the quant config, and the
# fused kernel applies them — but the NVFP4 quant-config plumbing in the
# middle drops them: modelopt's builder call passes only swiglu_limit,
# the oracle's make_nvfp4_moe_quant_config doesn't accept alpha/beta at
# all, and nvfp4_w4a16_moe_quant_config doesn't forward them. MarlinExperts
# then silently falls back to alpha=1.0 / beta=0.0 — wrong activation
# math on every expert -> fluent multilingual garbage output.
#
# (Root cause identified independently by baristankut on NVIDIA forum
# 376979, working M3 deployment on 4x DGX Spark; unreported upstream as
# of 2026-07-18. This mod threads the two params through the same three
# joints his fix named, mirroring the pattern quark_moe.py already uses.)
#
# Applied as exact-string edits, idempotent, refuses loudly on drift.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MODELOPT="$PYTHON_ROOT/vllm/model_executor/layers/quantization/modelopt.py"
ORACLE="$PYTHON_ROOT/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"
MOECFG="$PYTHON_ROOT/vllm/model_executor/layers/fused_moe/config.py"

for f in "$MODELOPT" "$ORACLE" "$MOECFG"; do
  if [ ! -f "$f" ]; then
    echo "[minimax-m3-swigluoai] target not found: $f" >&2
    exit 1
  fi
done

python3 - "$MODELOPT" "$ORACLE" "$MOECFG" <<'PY'
from pathlib import Path
import py_compile
import sys

def patch(path, old, new, label, require=None):
    text = path.read_text()
    if require is not None and require not in text:
        raise SystemExit(
            f"[minimax-m3-swigluoai] {path.name}: prerequisite {require!r} "
            f"missing; upstream layout changed, refusing to patch ({label})."
        )
    if new in text:
        print(f"[minimax-m3-swigluoai] {path.name}: {label} already applied.")
        return
    if text.count(old) != 1:
        raise SystemExit(
            f"[minimax-m3-swigluoai] {path.name}: expected exactly one "
            f"occurrence of the {label} anchor, found {text.count(old)}; "
            f"upstream code changed, refusing to patch."
        )
    path.write_text(text.replace(old, new))
    py_compile.compile(str(path), doraise=True)
    print(f"[minimax-m3-swigluoai] {path.name}: {label} applied.")

modelopt, oracle, moecfg = (Path(p) for p in sys.argv[1:4])

# 1. modelopt.py — pass the layer's swiglu alpha/beta into the NVFP4
#    quant-config builder (mirrors its existing swiglu_limit line and the
#    getattr pattern quark_moe.py uses).
patch(
    modelopt,
    old=(
        "        return make_nvfp4_moe_quant_config(\n"
        "            backend=self.nvfp4_backend,\n"
        "            w13_scale=layer.w13_weight_scale,\n"
        "            w2_scale=layer.w2_weight_scale,\n"
        "            w13_scale_2=layer.w13_weight_scale_2,\n"
        "            w2_scale_2=layer.w2_weight_scale_2,\n"
        "            a13_scale=layer.w13_input_scale,\n"
        "            a2_scale=layer.w2_input_scale,\n"
        "            swiglu_limit=getattr(layer, \"swiglu_limit\", None),\n"
        "            layer=layer,\n"
        "        )"
    ),
    new=(
        "        return make_nvfp4_moe_quant_config(\n"
        "            backend=self.nvfp4_backend,\n"
        "            w13_scale=layer.w13_weight_scale,\n"
        "            w2_scale=layer.w2_weight_scale,\n"
        "            w13_scale_2=layer.w13_weight_scale_2,\n"
        "            w2_scale_2=layer.w2_weight_scale_2,\n"
        "            a13_scale=layer.w13_input_scale,\n"
        "            a2_scale=layer.w2_input_scale,\n"
        "            swiglu_limit=getattr(layer, \"swiglu_limit\", None),\n"
        "            swiglu_alpha=getattr(layer, \"swiglu_alpha\", None),\n"
        "            swiglu_beta=getattr(layer, \"swiglu_beta\", None),\n"
        "            layer=layer,\n"
        "        )"
    ),
    label="builder-call params",
)

# 2a. oracle/nvfp4.py — accept the params in make_nvfp4_moe_quant_config.
patch(
    oracle,
    old=(
        "    a2_scale: torch.Tensor,\n"
        "    swiglu_limit: float | None = None,\n"
        "    layer: torch.nn.Module | None = None,\n"
        ") -> FusedMoEQuantConfig:"
    ),
    new=(
        "    a2_scale: torch.Tensor,\n"
        "    swiglu_limit: float | None = None,\n"
        "    swiglu_alpha: float | None = None,\n"
        "    swiglu_beta: float | None = None,\n"
        "    layer: torch.nn.Module | None = None,\n"
        ") -> FusedMoEQuantConfig:"
    ),
    label="builder signature",
    require="def make_nvfp4_moe_quant_config(",
)

# 2b. oracle/nvfp4.py — forward them in the MARLIN branch.
patch(
    oracle,
    old=(
        "    elif backend == NvFp4MoeBackend.MARLIN:\n"
        "        return nvfp4_w4a16_moe_quant_config(\n"
        "            g1_alphas=w13_scale_2,\n"
        "            g2_alphas=w2_scale_2,\n"
        "            w1_scale=w13_scale,\n"
        "            w2_scale=w2_scale,\n"
        "            gemm1_clamp_limit=swiglu_limit,\n"
        "        )"
    ),
    new=(
        "    elif backend == NvFp4MoeBackend.MARLIN:\n"
        "        return nvfp4_w4a16_moe_quant_config(\n"
        "            g1_alphas=w13_scale_2,\n"
        "            g2_alphas=w2_scale_2,\n"
        "            w1_scale=w13_scale,\n"
        "            w2_scale=w2_scale,\n"
        "            gemm1_clamp_limit=swiglu_limit,\n"
        "            gemm1_alpha=swiglu_alpha,\n"
        "            gemm1_beta=swiglu_beta,\n"
        "        )"
    ),
    label="MARLIN branch forwarding",
)

# 3. fused_moe/config.py — nvfp4_w4a16_moe_quant_config accepts and
#    forwards into FusedMoEQuantConfig.make (whose gemm1_alpha/gemm1_beta
#    fields already exist upstream).
patch(
    moecfg,
    old=(
        "def nvfp4_w4a16_moe_quant_config(\n"
        "    g1_alphas: torch.Tensor,\n"
        "    g2_alphas: torch.Tensor,\n"
        "    w1_scale: torch.Tensor,\n"
        "    w2_scale: torch.Tensor,\n"
        "    gemm1_clamp_limit: float | None = None,\n"
        ") -> FusedMoEQuantConfig:"
    ),
    new=(
        "def nvfp4_w4a16_moe_quant_config(\n"
        "    g1_alphas: torch.Tensor,\n"
        "    g2_alphas: torch.Tensor,\n"
        "    w1_scale: torch.Tensor,\n"
        "    w2_scale: torch.Tensor,\n"
        "    gemm1_clamp_limit: float | None = None,\n"
        "    gemm1_alpha: float | None = None,\n"
        "    gemm1_beta: float | None = None,\n"
        ") -> FusedMoEQuantConfig:"
    ),
    label="w4a16 config signature",
    require="gemm1_alpha: float | None = None",
)

patch(
    moecfg,
    old=(
        "    return FusedMoEQuantConfig.make(\n"
        "        quant_dtype=None,\n"
        "        w1_scale=w1_scale,\n"
        "        w2_scale=w2_scale,\n"
        "        g1_alphas=g1_alphas,\n"
        "        g2_alphas=g2_alphas,\n"
        "        weight_dtype=\"nvfp4\",\n"
        "        gemm1_clamp_limit=gemm1_clamp_limit,\n"
        "    )"
    ),
    new=(
        "    return FusedMoEQuantConfig.make(\n"
        "        quant_dtype=None,\n"
        "        w1_scale=w1_scale,\n"
        "        w2_scale=w2_scale,\n"
        "        g1_alphas=g1_alphas,\n"
        "        g2_alphas=g2_alphas,\n"
        "        weight_dtype=\"nvfp4\",\n"
        "        gemm1_clamp_limit=gemm1_clamp_limit,\n"
        "        gemm1_alpha=gemm1_alpha,\n"
        "        gemm1_beta=gemm1_beta,\n"
        "    )"
    ),
    label="w4a16 config forwarding",
)
PY

echo "=====> Marlin NVFP4 MoE now receives SwiGLU-OAI gemm1_alpha/gemm1_beta (MiniMax-M3 garbage fix)"
