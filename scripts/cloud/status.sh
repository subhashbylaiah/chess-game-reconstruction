#!/bin/zsh
# Run on the Mac: progress of every cloud run (step, speed, ETA, validation curve).
# Usage: scripts/cloud/status.sh ubuntu@<ip> [ubuntu@<ip2> ...]   (instances share the filesystem: runs listed once)
for HOST in "$@"; do
  ssh -o ConnectTimeout=10 $HOST 'echo "== $(hostname)  GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)  running: $(tmux ls 2>/dev/null | cut -d: -f1) $(pgrep -fa "recon train" | grep -o "runs/[a-z0-9_]*" | head -1)"'
done
ssh -o ConnectTimeout=10 $1 'cd /lambda/nfs/recon/recon 2>/dev/null || cd ~/recon
for d in runs/*/; do [ -f $d/log.jsonl ] || continue; python3 - $d <<"PY"
import json, sys, time, os
d = sys.argv[1].rstrip("/"); cfg = json.load(open(d + "/config.json")); total = cfg["steps"]
rows = [json.loads(l) for l in open(d + "/log.jsonl")]
tr = [r for r in rows if "train" in r]; va = [r for r in rows if "val" in r]
if not tr: sys.exit()
last = tr[-1]; step = last["step"]; sp = last["pos_per_s"]
age = time.time() - os.path.getmtime(d + "/log.jsonl")
eta = (total - step) * cfg["batch_size"] / sp / 3600
state = "done" if step >= total else ("running" if age < 600 else "stalled/stopped (no log for %.0f min)" % (age / 60))
print("%-18s step %6d/%d (%3.0f%%)  %5.0f pos/s  ETA %.1f h  [%s]" % (os.path.basename(d), step, total, 100 * step / total, sp, eta if step < total else 0, state))
if va: print("%18s val missing top-1/2/5: " % "" + "  ".join("%dk %.1f/%s/%.1f" % (r["step"] // 1000, 100 * r["val"]["hid@1"], ("%.1f" % (100 * r["val"]["hid@2"])) if "hid@2" in r["val"] else "-", 100 * r["val"]["hid@5"]) for r in va[-6:]))
PY
done'
