#!/usr/bin/env bash
# setup/download_model.sh
# Downloads the Nemotron-3-Super-120B-A12B NVFP4 weights to the local cache.
# Run this before docker/start.sh
set -euo pipefail

echo "=== Downloading Nemotron-3-Super NVFP4 Weights ==="

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "ERROR: NGC_API_KEY environment variable is not set."
  echo "Please set it using: export NGC_API_KEY='your_api_key'"
  exit 1
fi

NIM_CACHE="${NIM_CACHE:-$HOME/nim-cache}"
mkdir -p "$NIM_CACHE"

echo "Downloading to $NIM_CACHE..."
echo "This will download ~75GB of weights."
echo "TIP: If you are on an SSH session, consider running this inside tmux (e.g., tmux new -t nemotron)."

docker run -it --rm \
  --runtime=nvidia --gpus all \
  -v "$NIM_CACHE:/opt/nim/.cache" \
  -e NGC_API_KEY="$NGC_API_KEY" \
  nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:latest \
  download-to-cache --profiles 3b37a659a22c9390abe7b16aeb29c301c2e9c0e12e5b0fa76171681df31930e0

echo ""
echo "✓ Download complete. Model cached at $NIM_CACHE"
echo "Next: bash docker/start.sh"