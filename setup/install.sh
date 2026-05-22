#!/usr/bin/env bash
# setup/install.sh
# Prerequisites for DGX Spark · Nemotron-3-Super-120B production setup
# Run once on a fresh DGX Spark before starting vLLM.
set -euo pipefail

echo "=== DGX Spark Setup ==="

# ── 1. Disable swap permanently ───────────────────────────────────────────────
echo "[1/4] Disabling swap..."
sudo swapoff -a
# Comment out active swap entries in fstab to survive reboots.
# Match any whitespace-separated mount rows, not just lines with literal spaces.
sudo sed -ri '/^[[:space:]]*#/! s@^([[:space:]]*[^[:space:]#]+[[:space:]]+[^[:space:]]+[[:space:]]+swap[[:space:]].*)$@#\1@' /etc/fstab
free -h | grep Swap
echo "    Swap disabled."

# ── 2. Verify Docker is running ───────────────────────────────────────────────
echo "[2/4] Checking Docker..."
docker version --format 'Docker {{.Server.Version}}' || {
  echo "ERROR: Docker not running. Start Docker first."
  exit 1
}

# ── 3. Install uv (for repo-managed benchmark wrappers and scripts) ──────────
echo "[3/4] Installing uv..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.cargo/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ── 4. Pre-pull the vLLM image (optional, large download) ────────────────────
echo "[4/4] Pulling vLLM image (this may take a while)..."
echo "    Image: vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
read -rp "    Pull now? [y/N] " pull_now
if [[ "${pull_now,,}" == "y" ]]; then
  docker pull vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
fi

echo ""
echo "=== Setup complete ==="
echo "Next: bash setup/download_model.sh"
