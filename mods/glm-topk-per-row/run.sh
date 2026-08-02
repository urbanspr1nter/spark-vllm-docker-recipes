#!/bin/bash
set -euo pipefail

# GLM-5.2 (DSA) beyond-400K context on DGX Spark (GB10 / sm_121):
# route the indexer's DECODE top-k away from the persistent_topk kernel.
#
# persistent_topk uses cooperative CTA groups sized by context length:
# on GB10, num_sms x occupancy = 48 co-resident CTAs, and max_model_len
# 400000 lands exactly at 48 CTAs — 450000 needs 54 and fails at launch
# (topk.cu:136). The dispatch's else-branch, ops.top_k_per_row_decode
# (vLLM's own per-row kernel, same family the PREFILL path already uses
# for every token ever served), has no CTA/context coupling.
#
# This mod forces use_persistent_topk = False so decode top-k takes the
# per-row kernel, removing the 400K ceiling. Trade-off: persistent_topk
# exists because it is faster — expect some decode-speed cost (unmeasured;
# validate). No weights/attention-math changes: kernel dispatch only,
# results must remain exact (verify with canaries + needle at depth).
#
# STATUS: DRAFT — not yet validated on hardware. Test plan in README.md.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/model_executor/layers/sparse_attn_indexer.py"

if [ ! -f "$TARGET" ]; then
  echo "[glm-topk-per-row] sparse_attn_indexer.py not found at $TARGET" >&2
  exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import py_compile
import sys

path = Path(sys.argv[1])
text = path.read_text()

old = (
    "        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (\n"
    "            512,\n"
    "            1024,\n"
    "            2048,\n"
    "        )"
)
new = (
    "        # [glm-topk-per-row] persistent_topk's cooperative CTA count\n"
    "        # scales with context and exceeds GB10's 48-CTA budget past\n"
    "        # 400K ctx; force the per-row decode kernel instead.\n"
    "        use_persistent_topk = False"
)

if "ops.top_k_per_row_decode(" not in text:
    raise SystemExit(
        "[glm-topk-per-row] top_k_per_row_decode fallback not found in "
        "sparse_attn_indexer.py; upstream removed the else-branch this mod "
        "relies on. Refusing to patch."
    )

if new in text:
    print("[glm-topk-per-row] already applied; skipping.")
elif old in text:
    if text.count(old) != 1:
        raise SystemExit(
            f"[glm-topk-per-row] expected exactly one anchor occurrence, "
            f"found {text.count(old)}; refusing to patch."
        )
    path.write_text(text.replace(old, new))
    py_compile.compile(str(path), doraise=True)
    print("[glm-topk-per-row] decode top-k routed to per-row kernel (400K ceiling lifted).")
else:
    raise SystemExit(
        "[glm-topk-per-row] anchor not found; upstream dispatch changed. "
        "Re-derive the patch against the current sparse_attn_indexer.py."
    )
PY

echo "=====> DSA decode top-k forced to per-row kernel (persistent_topk 48-CTA ceiling bypassed)"
