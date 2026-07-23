#!/bin/bash
set -euo pipefail

# MiniMax-M3: fix the shared topk_indices_buffer layout mismatch in the
# Triton sparse path — THE root cause of the "any concurrency kills the
# engine" crash on GB10/sm_121 (vLLM dev1237).
#
# model.py allocates the buffer TOKEN-major ([padded_tokens, num_index
# _heads, topk] — correct for the SM100 MSA impl it was built alongside),
# but the platform-generic Triton indexer impl (common/indexer.py) and
# Triton attend (common/sparse_attention.py) slice it HEAD-major:
# buf[:, :nd], buf[:, nd:], topk[:, :nd, :], topk[:, nd:num_tokens, :].
# With one index head per rank (TP=4) the strides coincide (K, K, 1), so
# pure-decode and pure-prefill batches accidentally address the right
# bytes — bs=1 traffic works flawlessly. But in a MIXED decode+prefill
# batch (nd >= 1), buf[:, nd:, :] slices the size-1 HEAD dim -> a
# zero-numel tensor -> torch data_ptr() returns NULL -> the prefill
# top-k kernel (_topk_index_kernel) wild-stores at the null page ->
# Warp Illegal Address (verified twice by CUDA coredump at
# index_topk.py:284, grid (q,1,1)) -> poisoned context -> the engine
# dies with assorted async IMA / "misaligned address" / NCCL-surfaced
# errors ~60s later (collective-timeout detection lag). Mixed batches
# only arise with >1 request in flight -> "concurrency kills M3".
# SM100 runs the MSA impl (token-major native) and never hits this.
#
# Fix: hand the Triton impls a permute(1, 0, 2) VIEW of the buffer —
# head-major shape over the same stable storage (cudagraph-safe, correct
# for any head count; all kernels consume explicit strides).
#
# Applied as exact-string edits, idempotent, refuses loudly on drift.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
INDEXER="$PYTHON_ROOT/vllm/models/minimax_m3/common/indexer.py"
SPARSE="$PYTHON_ROOT/vllm/models/minimax_m3/common/sparse_attention.py"

for f in "$INDEXER" "$SPARSE"; do
  if [ ! -f "$f" ]; then
    echo "[minimax-m3-topk-buffer-layout] target not found: $f" >&2
    exit 1
  fi
done

python3 - "$INDEXER" "$SPARSE" <<'PY'
from pathlib import Path
import py_compile
import sys

def patch(path, old, new, label):
    text = path.read_text()
    if new in text:
        print(f"[minimax-m3-topk-buffer-layout] {path.name}: {label} already applied.")
        return
    if text.count(old) != 1:
        raise SystemExit(
            f"[minimax-m3-topk-buffer-layout] {path.name}: expected exactly "
            f"one occurrence of the {label} anchor, found {text.count(old)}; "
            f"upstream code changed, refusing to patch."
        )
    path.write_text(text.replace(old, new))
    py_compile.compile(str(path), doraise=True)
    print(f"[minimax-m3-topk-buffer-layout] {path.name}: {label} applied.")

indexer, sparse = Path(sys.argv[1]), Path(sys.argv[2])

# Triton indexer impl: head-major view of the token-major shared buffer.
patch(
    indexer,
    old=(
        "        # Both sides write into the single shared persistent topk_indices_buffer\n"
        "        # (decode at [:, :nd], prefill at [:, nd:]) and return views into it; the\n"
        "        # kernels' out= writes out[:, :total_q]. None -> allocate fresh.\n"
        "        buf = self.topk_indices_buffer"
    ),
    new=(
        "        # Both sides write into the single shared persistent topk_indices_buffer\n"
        "        # (decode at [:, :nd], prefill at [:, nd:]) and return views into it; the\n"
        "        # kernels' out= writes out[:, :total_q]. None -> allocate fresh.\n"
        "        # LAYOUT FIX (mods/minimax-m3-topk-buffer-layout): the shared buffer is\n"
        "        # allocated TOKEN-major [padded_tokens, heads, topk] (model.py, native\n"
        "        # for the SM100 MSA impl); this impl slices it HEAD-major. Slicing the\n"
        "        # size-1 head dim with nd >= 1 (mixed decode+prefill batch) yields a\n"
        "        # zero-numel tensor whose data_ptr() is NULL -> the prefill top-k\n"
        "        # kernel wild-stores at the null page and kills the engine. Use a\n"
        "        # permuted head-major VIEW (same stable storage, stride-addressed by\n"
        "        # every consumer kernel, correct for any head count).\n"
        "        buf = self.topk_indices_buffer\n"
        "        if buf is not None:\n"
        "            buf = buf.permute(1, 0, 2)"
    ),
    label="indexer head-major view",
)

# Triton attend impl: same view on the read side.
patch(
    sparse,
    old=(
        "        # Indexer top-k from the shared buffer: decode [:, :nd], prefill [:, nd:].\n"
        "        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]\n"
        "        assert topk is not None"
    ),
    new=(
        "        # Indexer top-k from the shared buffer: decode [:, :nd], prefill [:, nd:].\n"
        "        # LAYOUT FIX (mods/minimax-m3-topk-buffer-layout): token-major shared\n"
        "        # buffer, head-major consumers — read through the same permuted view\n"
        "        # the indexer writes through (see common/indexer.py for the analysis).\n"
        "        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]\n"
        "        assert topk is not None\n"
        "        topk = topk.permute(1, 0, 2)"
    ),
    label="attend head-major view",
)
PY

echo "=====> MiniMax-M3 topk buffer layout fixed (mixed decode+prefill batches no longer dereference NULL)"
