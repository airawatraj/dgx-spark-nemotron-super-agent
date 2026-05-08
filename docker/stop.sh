#!/usr/bin/env bash
# docker/stop.sh
# Stops and removes the spark-brain container.
set -euo pipefail

if docker ps --format '{{.Names}}' | grep -q '^spark-brain$'; then
  echo "Stopping spark-brain..."
  docker stop spark-brain
  docker rm spark-brain
  echo "✓ spark-brain stopped and removed."
else
  echo "spark-brain is not running."
  docker rm spark-brain 2>/dev/null && echo "Removed stopped container." || true
fi