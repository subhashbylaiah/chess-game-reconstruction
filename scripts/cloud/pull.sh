#!/bin/zsh
# Run on the Mac: copy finished runs (logs, checkpoints) back from the cloud machine into runs/.
# Usage: scripts/cloud/pull.sh ubuntu@<ip> [path/to/key.pem]
set -e
HOST=${1:?usage: pull.sh user@host [key.pem]}; KEY=${2:-}
DIR=${RECON_DIR:-/lambda/nfs/recon/recon}
SSH=(ssh); [[ -n $KEY ]] && SSH=(ssh -i $KEY)
cd "$(dirname $0)/../.."
rsync -az -e "${SSH[*]}" --exclude 'state.pt' --exclude 'speedtest*' $HOST:$DIR/runs/ runs/
