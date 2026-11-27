#!/bin/bash
# Measure training throughput for a few data-worker counts (about 5 min). Pick the fastest for queue.sh.
cd "$(dirname "$0")/../.."; source ~/.local/bin/env 2>/dev/null
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/.venv-recon}  # venv on local disk, not the network filesystem
mkdir -p runs
N=$(nproc)
for W in 8 16 $((N > 4 ? N - 2 : N)); do
  [[ $W -gt $N ]] && continue
  rm -rf runs/speedtest_$W
  uv run recon train --out runs/speedtest_$W --steps 400 --log-every 100 --eval-every 100000 \
      --val-games 20 --workers $W --amp --no-resume > runs/speedtest_$W.out 2>&1
  echo "workers=$W  $(grep '^step 400' runs/speedtest_$W.out | grep -o '[0-9]* pos/s')"
done
echo "(Mac M2 Pro reference: about 750 pos/s with 6 workers)"
