#!/usr/bin/env bash
# setup/download_parser.sh
# Downloads the super_v3 reasoning parser from NVIDIA's official HuggingFace repo.
# Must be run from the repo root (where docker/start.sh expects to find it).
set -euo pipefail

PARSER_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py"
DEST="$(dirname "$0")/../super_v3_reasoning_parser.py"

echo "Downloading super_v3_reasoning_parser.py from NVIDIA HuggingFace..."

# Pass HF_TOKEN if available (required if the model repo becomes gated)
WGET_OPTS=(-q --show-progress)
if [[ -n "${HF_TOKEN:-}" ]]; then
  WGET_OPTS+=(--header "Authorization: Bearer ${HF_TOKEN}")
fi

wget "${WGET_OPTS[@]}" -O "$DEST" "$PARSER_URL"

# ── Validate the download ─────────────────────────────────────────────────────
# HuggingFace can return a 200 OK HTML redirect page on auth failure or repo
# restructuring. The `file` command reports "ASCII text" for both HTML and Python,
# so we inspect the content directly instead.
#
# Strategy: check the first 10 lines for at least one Python keyword.
# An HTML error page will contain "<!DOCTYPE", "<html>", or "Redirecting" instead.

if head -10 "$DEST" | grep -qE '^\s*(import |from |def |class |#)'; then
  echo "✓ Downloaded and validated: $DEST"
else
  echo "ERROR: Downloaded file does not look like Python source."
  echo "       It may be an HTML redirect or auth error from HuggingFace."
  echo "       If the repo is gated, set HF_TOKEN and retry."
  echo ""
  echo "First 10 lines of downloaded file:"
  head -10 "$DEST"
  rm -f "$DEST"
  exit 1
fi

echo "Next: bash docker/start.sh"