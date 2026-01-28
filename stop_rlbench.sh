#!/usr/bin/env bash
set -euo pipefail
docker rm -f rlbench >/dev/null 2>&1 || true
sudo pkill -f "Xorg :99" >/dev/null 2>&1 || true
echo "Stopped container and Xorg :99"
