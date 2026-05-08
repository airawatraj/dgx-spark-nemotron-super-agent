#!/usr/bin/env bash
# setup/download_parser.sh
# Downloads the super_v3 reasoning parser from NVIDIA's official HuggingFace repo.
# Must be run from the repo root (where docker/start.sh expects to find it).
set -euo pipefail

PARSER_URL="https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py"
DEST="$(dirname "$0")/../super_v3_reasoning_parser.py"

echo "Downloading super_v3_reasoning_parser.py from NVIDIA HuggingFace..."
wget -q --show-progress -O "$DEST" "$PARSER_URL"

# Verify it's a Python file, not a redirect/error page
if file "$DEST" | grep -q "Python"; then
  echo "✓ Downloaded: $DEST"
elif head -1 "$DEST" | grep -q "def\|import\|class"; then
  echo "✓ Downloaded: $DEST"
else
  echo "ERROR: Downloaded file does not look like Python source."
  echo "       Check your HF_TOKEN or network access and retry."
  cat "$DEST" | head -5
  exit 1
fi

echo "Next: bash docker/start.sh"
