#!/usr/bin/env bash
# docker/status.sh
# Shows the health of the spark-brain container, system memory, and VmSwap.
set -euo pipefail

echo "=== spark-brain status ==="
echo ""

# ── Container ─────────────────────────────────────────────────────────────────
if docker ps --format '{{.Names}}' | grep -q '^spark-brain$'; then
  UPTIME=$(docker inspect spark-brain --format '{{.State.StartedAt}}')
  echo "  Container:  running (started $UPTIME)"
else
  echo "  Container:  NOT RUNNING"
fi

# ── vLLM health ───────────────────────────────────────────────────────────────
if curl -sf http://localhost:8000/health &>/dev/null; then
  echo "  vLLM API:   healthy (http://localhost:8000)"
else
  echo "  vLLM API:   not reachable"
fi

# ── Memory ────────────────────────────────────────────────────────────────────
echo ""
echo "=== System memory ==="
free -h

# ── Swap for vLLM process ─────────────────────────────────────────────────────
echo ""
echo "=== vLLM VmSwap ==="
PID=$(docker inspect --format '{{.State.Pid}}' spark-brain 2>/dev/null || echo "")
if [[ -n "$PID" && "$PID" != "0" ]]; then
  VMSWAP=$(grep VmSwap /proc/$PID/status 2>/dev/null || echo "VmSwap: unavailable")
  echo "  $VMSWAP"
  if echo "$VMSWAP" | grep -q "0 kB"; then
    echo "  ✓ No swap in use"
  else
    echo "  ⚠ WARNING: vLLM is using swap — performance will be degraded"
  fi
else
  echo "  Container not running or PID unavailable"
fi

# ── KV cache usage ────────────────────────────────────────────────────────────
echo ""
echo "=== KV cache ==="
METRICS=$(curl -sf http://localhost:8000/metrics 2>/dev/null || echo "")
if [[ -n "$METRICS" ]]; then
  KV=$(printf '%s\n' "$METRICS" | awk '!/^#/ && /gpu_cache_usage_perc/ {printf "%.1f%%", $2 * 100; exit}')
  RUNNING=$(printf '%s\n' "$METRICS" | awk '!/^#/ && /(^|:)num_requests_running([[:space:]]|$)/ {print $2; exit}')
  echo "  KV cache used:     ${KV:-unknown}"
  echo "  Requests running:  ${RUNNING:-0}"
else
  echo "  Metrics endpoint not available"
fi

# ── Last 5 log lines ──────────────────────────────────────────────────────────
echo ""
echo "=== Recent logs ==="
if docker ps -a --format '{{.Names}}' | grep -q '^spark-brain$'; then
  docker logs spark-brain --tail 5 2>&1 | sed 's/^/  /'
else
  echo "  No spark-brain container found"
fi
