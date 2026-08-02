#!/bin/bash
# Fix DSA indexer expanded_block_table_buffer off-by-one with MTP at
# block-aligned max_model_len (vLLM issue #46074).
#
# The indexer sizes the buffer as cdiv(max_model_len, block_size * shards),
# but MTP speculative tokens can extend a request up to num_speculative_tokens
# past max_model_len. When max_model_len is an exact multiple of block_size
# (ours: 400000 / 64 = 6250, no slack), the decode path needs one more block
# than the buffer holds -> worker RuntimeError ("expanded size of the tensor
# (N) must match the existing size (N+1)") at ceiling-depth contexts.
#
# +1 block of headroom covers any num_speculative_tokens <= block_size.
# Same fix as tonyd2wild's fix-indexer-mtp-overhang.py, re-anchored for the
# dev1479 code layout. Drop when #46074 is fixed upstream.
#
# Idempotent: exits 0 without change if already applied; fails loudly if the
# anchor is missing (upstream layout changed - re-verify before trusting).
set -euo pipefail

FILE=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py

MARKER="MTP spec tokens can extend a request one block past max_model_len"
if grep -q "$MARKER" "$FILE"; then
    echo "[glm-indexer-mtp-overhang] already applied."
    echo "=====> DSA indexer MTP-overhang buffer fix present"
    exit 0
fi

OLD='            self.kv_cache_spec.block_size * get_kv_cache_shard_count(),
        )'
NEW='            self.kv_cache_spec.block_size * get_kv_cache_shard_count(),
        ) + 1  # MTP spec tokens can extend a request one block past max_model_len'

if ! grep -qF "get_kv_cache_shard_count()," "$FILE"; then
    echo "[glm-indexer-mtp-overhang] ERROR: anchor not found in $FILE" >&2
    echo "[glm-indexer-mtp-overhang] upstream layout changed - re-verify the fix" >&2
    exit 1
fi

python3 - "$FILE" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = """            self.kv_cache_spec.block_size * get_kv_cache_shard_count(),
        )
        self.expanded_block_table_buffer = torch.zeros("""
new = """            self.kv_cache_spec.block_size * get_kv_cache_shard_count(),
        ) + 1  # MTP spec tokens can extend a request one block past max_model_len
        self.expanded_block_table_buffer = torch.zeros("""
if old not in src:
    sys.exit("[glm-indexer-mtp-overhang] ERROR: exact anchor block not found")
open(path, "w").write(src.replace(old, new, 1))
print("[glm-indexer-mtp-overhang] patched expanded_block_table_buffer sizing (+1 block).")
EOF

echo "=====> DSA indexer MTP-overhang buffer fix applied (vLLM #46074)"
