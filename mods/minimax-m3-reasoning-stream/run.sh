#!/bin/bash
set -euo pipefail

# MiniMax-M3: stop the streaming reasoning leak (<mm:think> in content).
#
# Root cause (2026-07-22, spark quartet): every serving-layer caller of
# ReasoningParser.is_reasoning_end passes PROMPT token ids only, as a gate
# for "did this turn's reasoning already finish inside the prompt?"
# (chat_completion/serving.py:353, parser/abstract_parser.py parse_delta
# prompt check, v1/structured_output/__init__.py:366). M3's bare marker
# scan misfires on EVERY prompt, twice over:
#   1. the chat template's <thinking_instructions> system block contains
#      "<mm:think></mm:think>" as literal instruction text, and both
#      markers are single-token vocab entries (200059/200060), so every
#      tokenized prompt ends with an end-marker after a start-marker;
#   2. multi-turn history preserves earlier turns' closed think blocks
#      (M3 interleaved thinking), same outcome.
# Result: state.reasoning_ended latches True on the first streamed delta,
# the reasoning phase is never entered, and the whole generation streams
# as raw content — "<mm:think>..." and all (vLLM #46042). Non-streaming
# is unaffected because the full generator parses only the output text.
#
# Fix: a generation prompt always ends at a fresh assistant turn, so the
# honest prompt-gate answer for M3 is always False ("not ended") — the
# text-marker streaming extractor then classifies generated text
# correctly whether the model thinks or not. Unreported upstream as of
# 2026-07-22 (last upstream touch to this file: #45718, 06-23).
#
# Applied as an exact-string edit, idempotent, refuses loudly on drift.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
PARSER="$PYTHON_ROOT/vllm/reasoning/minimax_m3_reasoning_parser.py"

if [ ! -f "$PARSER" ]; then
  echo "[minimax-m3-reasoning-stream] target not found: $PARSER" >&2
  exit 1
fi

python3 - "$PARSER" <<'PY'
from pathlib import Path
import py_compile
import sys

def patch(path, old, new, label, require=None):
    text = path.read_text()
    if require is not None and require not in text:
        raise SystemExit(
            f"[minimax-m3-reasoning-stream] {path.name}: prerequisite "
            f"{require!r} missing; upstream layout changed, refusing to "
            f"patch ({label})."
        )
    if new in text:
        print(f"[minimax-m3-reasoning-stream] {path.name}: {label} already applied.")
        return
    if text.count(old) != 1:
        raise SystemExit(
            f"[minimax-m3-reasoning-stream] {path.name}: expected exactly "
            f"one occurrence of the {label} anchor, found "
            f"{text.count(old)}; upstream code changed, refusing to patch."
        )
    path.write_text(text.replace(old, new))
    py_compile.compile(str(path), doraise=True)
    print(f"[minimax-m3-reasoning-stream] {path.name}: {label} applied.")

parser = Path(sys.argv[1])

patch(
    parser,
    old=(
        "    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n"
        "        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)\n"
        "        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)\n"
        "        if end_index < 0:\n"
        "            return False\n"
        "        if start_index < 0:\n"
        "            return True\n"
        "        return end_index > start_index"
    ),
    new=(
        "    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:\n"
        "        # Prompt-state gate: every call site passes prompt token ids\n"
        "        # only. M3 prompts ALWAYS contain both markers — as literal\n"
        "        # instruction text in <thinking_instructions> (single-token\n"
        "        # vocab entries) and as preserved think blocks in multi-turn\n"
        "        # history — so a bare marker scan reports \"reasoning already\n"
        "        # ended\", which disables streaming reasoning extraction and\n"
        "        # leaks <mm:think> into streamed content (vLLM #46042). A\n"
        "        # generation prompt always ends at a fresh assistant turn:\n"
        "        # reasoning there has NOT ended. The text-marker streaming\n"
        "        # extractor classifies generated text correctly from there,\n"
        "        # with or without a think block.\n"
        "        return False"
    ),
    label="prompt-gate fix",
    require="class MiniMaxM3ReasoningParser",
)
PY

echo "=====> MiniMax-M3 streaming reasoning prompt-gate fixed (no more <mm:think> leak in streamed content)"
