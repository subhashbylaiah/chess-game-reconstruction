#!/bin/zsh
# Terminate one Lambda instance (matched by IP) once a watch file shows a pattern, or at a deadline.
# Usage: terminate_after.sh <ip> <watch-file> <pattern> <deadline: unix epoch seconds>. Key: ~/.config/lambda/key (never printed).
IP=$1; WATCH=$2; PAT=$3; DEADLINE=$4
KEY=$(cat ~/.config/lambda/key); API=https://cloud.lambda.ai/api/v1
until grep -q "$PAT" "$WATCH" 2>/dev/null || (( $(date +%s) >= DEADLINE )); do sleep 30; done
echo "trigger: $(grep -q "$PAT" "$WATCH" 2>/dev/null && echo pattern || echo deadline) at $(date -u +%H:%M)"
ID=$(curl -s -u "$KEY:" $API/instances | python3 -c "import json,sys; print(next((i['id'] for i in json.load(sys.stdin)['data'] if i.get('ip')=='$IP'), ''))")
[[ -z $ID ]] && { echo "no instance with ip $IP (already gone?)"; exit 0; }
curl -s -u "$KEY:" -H 'Content-Type: application/json' -d "{\"instance_ids\": [\"$ID\"]}" $API/instance-operations/terminate | python3 -c "import json,sys; d=json.load(sys.stdin); print('terminated:', [i['id'] for i in d.get('data',{}).get('terminated_instances',[])] or d)"
