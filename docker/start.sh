#!/usr/bin/env bash
# docker/start.sh
# Launches Nemotron-3-Super-120B on a single DGX Spark.
# Run from the repo root after completing setup/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARSER="$REPO_ROOT/super_v3_reasoning_parser.py"
NIM_CACHE="${NIM_CACHE:-$HOME/nim-cache}"

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "=== spark-brain preflight ==="

if [[ ! -f "$PARSER" ]]; then
  echo "ERROR: $PARSER not found. Run setup/download_parser.sh first."
  exit 1
fi

if [[ ! -d "$NIM_CACHE" ]]; then
  echo "ERROR: Model cache not found at $NIM_CACHE"
  echo "       Set NIM_CACHE env var to your model cache directory."
  exit 1
fi

MODEL_PATH=$(find "$NIM_CACHE" -type d -name "rl-030326-nvfp4" 2>/dev/null | head -1)
if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: rl-030326-nvfp4 snapshot not found in $NIM_CACHE"
  echo "       Ensure the NVFP4 model snapshot is present."
  exit 1
fi

SWAP=$(free | awk '/^Swap:/ {print $2}')
if [[ "$SWAP" -gt 0 ]]; then
  echo "WARNING: Swap is enabled ($SWAP kB). Disabling permanently now..."
  sudo swapoff -a
  sudo sed -i '/ swap / s/^\(.*\)$/#\1/' /etc/fstab
  echo "✓ OS Swap disabled and removed from /etc/fstab."
fi

echo "    Model:  $MODEL_PATH"
echo "    Parser: $PARSER"
echo "    Cache:  $NIM_CACHE"
echo ""

# ── Remove existing container if present ─────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q '^spark-brain$'; then
  echo "Removing existing spark-brain container..."
  docker stop spark-brain 2>/dev/null || true
  docker rm spark-brain 2>/dev/null || true
fi

# ── Build optional env flags ──────────────────────────────────────────────────
# NGC_API_KEY is only passed if set in the calling environment.
# Passing an empty string via -e NGC_API_KEY="" can cause unexpected behaviour
# in some Docker versions (treats it as a literal empty var rather than omitting it).
NGC_ENV=()
if [[ -n "${NGC_API_KEY:-}" ]]; then
  NGC_ENV=(-e "NGC_API_KEY=${NGC_API_KEY}")
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# Note on --tool-call-parser qwen3_coder:
# Routes tool calls without tripping the broken internal chat serving stub in the
# NVIDIA vLLM fork. Confirmed on vLLM 0.19.2rc1.dev134+gfe9c3d6c5
# (nightly image sha256:3dbe092e). Behaviour may differ on other builds — see METHODOLOGY.md.
echo "Starting spark-brain..."
docker run -d --name spark-brain --gpus all \
  --restart=unless-stopped \
  --shm-size=16gb \
  -p 8000:8000 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_HUB_OFFLINE=1 \
  "${NGC_ENV[@]}" \
  -v "$NIM_CACHE:/nim-cache" \
  -v "$PARSER:/app/super_v3_reasoning_parser.py" \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
    --model "$MODEL_PATH" \
    --served-model-name Cogni-Brain \
    --host 0.0.0.0 --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 4 \
    --max-model-len 131072 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

echo ""
echo "Container started. Waiting for server to be ready (~10 minutes)..."
echo "Watch progress: docker logs -f spark-brain"
echo ""

# Wait for ready signal
timeout 900 bash -c '
  until curl -sf http://localhost:8000/health &>/dev/null; do
    sleep 5
  done
' && echo "✓ Server ready at http://localhost:8000" || echo "Timeout — check docker logs spark-brain"