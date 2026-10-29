#!/usr/bin/env bash
# Evaluate a learned-policy weights dir (<task>.pt, optional text/) on one task in the official
# loop: one reused env per variant, seeds in sequence, like the official eval.
#   dev/eval_lpolicy.sh WEIGHTS_DIR TASK OUT.jsonl [N_SEEDS] [EXTRA_PROFILES] [FIRST_SEED]
# Every variant runs under "clean" (Understanding / Behavior); each extra profile (e.g.
# "light_l3 cam_l3") runs the default variant only, as the Perception sweep does.
# Seeds FIRST_SEED .. FIRST_SEED + N_SEEDS - 1 (default 2,700,001). ORACLE_VARIANT=1 bypasses the
# text classifier; LPOLICY_EXEC sets the actions executed per chunk; SKIP_CLEAN=1 runs only the
# extra profiles. GPU_ID picks the GPU (default 0); MF_RENDER_BACKEND=cpu renders on the CPU.
set -u
ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
W=$(readlink -f "$1"); TASK=$2; OUT=$(readlink -f "$3"); N=${4:-20}; PROFILES=${5:-}; FIRST=${6:-2700001}
SEEDS=$(seq $FIRST $((FIRST + N - 1)))
export OMP_NUM_THREADS=4  # the in-process policy runs small CPU ops; keep torch from spinning a thread per core
VARIANTS=$(python -c "
import sys; sys.path.insert(0, '$ROOT')
from dev.simenv import load_cfg; print(' '.join(load_cfg('$TASK')['variants']))")
DEFAULT=$(python -c "
import sys; sys.path.insert(0, '$ROOT')
from dev.simenv import load_cfg; print(load_cfg('$TASK')['perception']['default_variant'])")
run() {  # profile variant
  POLICY=lpolicy LPOLICY_WEIGHTS=$W PROFILE=$1 RESULTS_JSONL=$OUT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-${GPU_ID:-0}} \
    python "$ROOT/dev/official_loop.py" "$TASK" "$2" $SEEDS 2>&1 | grep -E "^TOTAL|Traceback|Error" || true
}
[ "${SKIP_CLEAN:-0}" = 1 ] || for v in $VARIANTS; do run clean "$v"; done
for p in $PROFILES; do run "$p" "$DEFAULT"; done
python - "$OUT" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["profile"], r["variant"])].append(r["success"])
for (p, v), s in sorted(agg.items()):
    print(f"{p:9s} {v:8s} {sum(s)}/{len(s)}")
clean = [r["success"] for r in rows if r["profile"] == "clean"]
print(f"clean, all variants: {sum(clean)}/{len(clean)}")
PY
