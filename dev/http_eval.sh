#!/usr/bin/env bash
# Serve a built submission folder ($SUBMISSION, default hf_submission_l) over HTTP and run one official
# eval script against it, as the organizers do.
# usage: [SUBMISSION=dir] dev/http_eval.sh <eval_module> <seeds.json> <out_dir> [n_seeds] [mode] [port]
#   e.g. dev/http_eval.sh eval.eval_insert_letter $PWD/seeds.json $PWD/runs/http_t5 3
# All paths must be absolute (the script changes directory).
set -u
MOD=$1; SEEDS=$2; OUT=$3; N=${4:-3}; MODE=${5:-both}; PORT=${6:-18080}
ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
REPO=${METAFINE_REPO:-$ROOT/MetaFine-Challenge_CoRL}
mkdir -p "$OUT"
cd "${SUBMISSION:-$ROOT/hf_submission_l}"
python policy_server.py --port "$PORT" > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 1; done
cd "$REPO"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m "$MOD" --policy-url "http://127.0.0.1:$PORT" --seeds "$SEEDS" \
  --n-seeds "$N" --mode "$MODE" --record-dir "$OUT/record" > "$OUT/eval.log" 2>&1
echo "eval exit code $?" >> "$OUT/eval.log"
