#!/bin/zsh
# Run on the Mac: copy the project (code, data, benchmark) to the cloud machine.
# Usage: scripts/cloud/push.sh ubuntu@<ip> [path/to/key.pem]
set -e
HOST=${1:?usage: push.sh user@host [key.pem]}; KEY=${2:-}
DIR=${RECON_DIR:-/lambda/nfs/recon/recon}  # persistent filesystem on Lambda; set RECON_DIR=~/recon elsewhere
SSH=(ssh); [[ -n $KEY ]] && SSH=(ssh -i $KEY)
cd "$(dirname $0)/../.."
rsync -az -e "${SSH[*]}" \
  --exclude .venv --exclude runs --exclude preds --exclude .git --exclude '__pycache__' --exclude '.codegraph' --exclude data/hcs --exclude data/broadcast \
  ./ $HOST:$DIR/
echo "pushed. next: ssh in, then: cd $DIR && bash scripts/cloud/setup.sh"
