# Training on a cloud GPU (AWS)

**Machine:** g6.8xlarge (1× L4, 32 vCPU) or g5.8xlarge (1× A10G, 32 vCPU), region us-east-2 (where the quota was requested).
**AMI:** "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (Ubuntu). **Disk:** 100 GB gp3. **Security group:** SSH from your IP only.

1. On the Mac: `scripts/cloud/push.sh ubuntu@<ip> ~/.ssh/<key>.pem`
2. On the machine: `ssh -i ~/.ssh/<key>.pem ubuntu@<ip>`, then `cd ~/recon && bash scripts/cloud/setup.sh`
3. `bash scripts/cloud/speedtest.sh`: note the worker count with the highest pos/s.
4. `tmux new -s train 'bash scripts/cloud/queue.sh <workers>'`. Detach with Ctrl-b d; reattach with `tmux attach -t train`.
   Progress: `tail -f runs/c13_long.out`.
5. If the machine is stopped or interrupted: start it again and rerun step 4. Runs resume from their last checkpoint
   (saved every 2,000 steps and at each eval) and finished runs are skipped.
6. On the Mac: `scripts/cloud/pull.sh ubuntu@<ip> ~/.ssh/<key>.pem` to bring runs back.
7. **Stop the instance when done**; it bills by the hour while running.

Notes:
- Cloud runs use `--amp` (bf16 on L4/A10G). Compare cloud runs only with cloud runs. `c13_long` is the cloud baseline for E13.
- A resumed run draws fresh training data (stream reseeded by step), so it doesn't replay the positions it already saw.
