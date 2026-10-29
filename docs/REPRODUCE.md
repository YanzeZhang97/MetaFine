# Reproducing the submitted networks

These are the exact data-generation and training commands behind the five control networks and the
instruction classifier of our final submission. Run them from the repository root, after the setup in
the main README.

- Each generation command writes one `.npz` file per successful episode, named `<variant>_<seed>.npz`, plus a
  `log.jsonl` with one line per attempted episode. Seeds `FIRST .. FIRST + N - 1` are run under every
  variant unless `--variants` is given.
- All training seeds are 5,000,000 or above, disjoint from the evaluation seeds (2,000,000 to 3,000,000 in our
  experiments).
- We ran the commands of a group as parallel jobs. Each simulator process needs about 1–2 GB of GPU memory, or
  none with `MF_RENDER_BACKEND=cpu CUDA_VISIBLE_DEVICES=""`; the hand-camera images match either way.
- `--set NAME=value` switches on a teacher option. Every option is documented where it is defined, in
  `dev/teacher.py` (`_GraspPart`, `_ToggleSwitch`, `_InsertLetter`, ...).
- Training ran on one RTX 5000 Ada GPU (about 5 GB of memory): 23–54 min per T1–T4 network, about 2 h for
  t5_g plus 30 min for each T5 fine-tuning round.
- Takeover rounds depend on the student that drove them, so later rounds reproduce only approximately unless
  every earlier step is repeated exactly. Simulation is deterministic per seed for T5; rendering noise makes
  some T1–T4 episodes differ between runs.
- `dev/gen_takeover.py --student` and `dev/eval_lpolicy.sh` take a weights folder holding `<task>.pt` and the
  instruction classifier in `text/`. After training a network, link one up with this helper (used below):

```bash
link_weights() {  # run task
  mkdir -p runs/$1/weights && ln -sf ../last.pt runs/$1/weights/$2.pt && ln -sfn ../../text_clf_v1 runs/$1/weights/text
}
```

- Merged training sets are folders of symbolic links to the episode files (seeds never repeat across rounds, so
  file names do not collide):

```bash
link_data() {  # dest files...
  d=data/demos/$1; shift; mkdir -p $d
  for f in "$@"; do ln -sf "$(readlink -f "$f")" "$d/"; done
}
```

## Instruction classifier (`text`)

```bash
python dev/gen_instructions.py data/instructions_train.jsonl
python dev/train_text.py data/instructions_train.jsonl runs/text_clf_v1   # needs data/models/all-MiniLM-L6-v2
```

`dev/train_text.py` reports accuracy on the 14 official instructions and on the 226 held-out paraphrases in
`dev/test_instructions.py`, which are never used for training.

## T2 `grasp_move_mug` (t2_c, 1,192 episodes)

```bash
for k in 0 1 2; do
  python dev/gen_demos.py grasp_move_mug data/demos/grasp_move_mug_blend $((5400000 + k * 134)) 134 --set DIRECT=true --set BLEND=true
done
python dev/train_act.py grasp_move_mug data/demos/grasp_move_mug_blend runs/t2_c --steps 40000 --goal
```

## T3 `toggle_switch_table` (t3_d, 1,955 episodes)

Teacher options from Sec. 3.2 of the report: capped settles, fingers closed while moving, a randomized
pre-push distance and lateral jitter.

```bash
for k in $(seq 0 19); do
  python dev/gen_demos.py toggle_switch_table data/demos/toggle_switch_table_v2 $((6000000 + k * 50)) 50 \
    --set SETTLE_CAP=3 --set NO_CLOSE_HOLD=true --set PRE_RANGE=[0.025,0.07] --set PRE_JITTER=0.004
done
python dev/train_act.py toggle_switch_table data/demos/toggle_switch_table_v2 runs/t3_d --steps 40000 --goal
```

## T4 `put_blocks_into_boxes` (t4_a, 896 episodes)

```bash
python dev/gen_demos.py put_blocks_into_boxes data/demos/put_blocks_into_boxes 5000000 300 --img-stride 2
python dev/train_act.py put_blocks_into_boxes data/demos/put_blocks_into_boxes runs/t4_a --steps 60000 --goal
```

## T1 `grasp_part` (t1_g, 1,410 episodes)

Plain demonstrations. The second command re-generates the body episodes of the first with a revised teacher
and replaces them; the last group stages deliberate misses (masked) followed by recoveries.

```bash
python dev/gen_demos.py grasp_part data/demos/grasp_part 5000000 400
python dev/gen_demos.py grasp_part data/demos/grasp_part 5000000 400 --variants body --set SMOOTH_SIDE=true --set EXT_LYING=true
python dev/train_act.py grasp_part data/demos/grasp_part runs/t1_b --steps 40000 --goal      # student for body takeovers
link_weights t1_b grasp_part
for k in 0 1 2; do
  python dev/gen_demos.py grasp_part data/demos/grasp_part $((5200000 + k * 84)) 84 --variants cap --set MISS_P=1.0 --set EXT_LYING=true
done
python dev/train_act.py grasp_part data/demos/grasp_part runs/t1_c --steps 40000 --goal      # student for cap takeovers
link_weights t1_c grasp_part
```

Takeovers, then the final network on both sets (`grasp_part_all` holds links to every file of `grasp_part` and
`grasp_part_to`):

```bash
for k in 0 1 2; do
  python dev/gen_takeover.py grasp_part data/demos/grasp_part_to $((5500000 + k * 100)) 100 --variants cap \
    --student runs/t1_c/weights --set EXT_LYING=true
  python dev/gen_takeover.py grasp_part data/demos/grasp_part_to $((5510000 + k * 100)) 100 --variants body \
    --student runs/t1_b/weights --set EXT_LYING=true --set SMOOTH_SIDE=true
done
mkdir -p data/demos/grasp_part_all
for f in data/demos/grasp_part/*.npz; do ln -sf "$(readlink -f "$f")" data/demos/grasp_part_all/; done
for f in data/demos/grasp_part_to/*.npz; do ln -sf "$(readlink -f "$f")" "data/demos/grasp_part_all/to_$(basename "$f")"; done
python dev/train_act.py grasp_part data/demos/grasp_part_all runs/t1_g --steps 60000 --goal
```

## T5 `insert_letter` (t5_j, 9,674 episodes)

T5 went through eight rounds. Each round changed the teacher (`--set` options) and/or added takeovers from the
previous student. All T5 networks use `--goal --crop --grip-mode reg`.

| Round | Data | Seeds | Student | Network trained after it |
|---|---|---|---|---|
| v1 | plain demos | 5,000,000–399 | – | t5_a (`insert_letter`, 60k) |
| v2 + r1 | teacher V2 demos; takeovers | 5,100,000–599; 5,300,000–299 | t5_a | t5_b (`insert_letter_v2`, 60k) |
| r2 | V2 + descent noise + misalignment demos; takeovers | 5,600,000–209; 5,310,000–299 | t5_b | t5_c (`insert_letter_c`, 80k) |
| r3 | takeovers | 5,320,000–299 | t5_c | t5_d, t5_d2 (`insert_letter_d`, 80k, two training runs) |
| v3 + r5 | V3 demos (capped settles, quick release); takeovers | 5,920,000–499; 5,930,000–249, 5,940,000–249 | t5_d, t5_d2 | t5_f (`insert_letter_f`, 80k) |
| r6 | takeovers resuming at the current height | 5,950,000–799 | t5_f | t5_g (`insert_letter_g`, 80k) |
| r7 | takeovers; release only once seated | 6,100,000–799 | t5_g | t5_h (fine-tuned from t5_g, 30k) |
| r8 | takeovers | 6,101,000–799 | t5_h | **t5_j** (fine-tuned from t5_h, 30k) |

Merged training sets (folders of symbolic links):

- `insert_letter_c`: the V2 demos with seeds 5,100,000–249, the r1 takeovers and all of r2 (4,067 episodes).
- `insert_letter_d`: the same V2 demos, all of r2 and all of r3 (4,013).
- `insert_letter_f`: v3 and r5 (3,818).
- `insert_letter_g`: v3, r6, and r5 without the "near" resumes whose TCP rose more than 3 mm in their first 12
  steps (5,358).
- `insert_letter_h`: `insert_letter_g` and r7 (7,444). `insert_letter_j`: `insert_letter_h` and r8 (9,674).

Commands (each group was run as parallel jobs):

```bash
# v1 -> t5_a
python dev/gen_demos.py insert_letter data/demos/insert_letter 5000000 400 --img-stride 2
python dev/train_act.py insert_letter data/demos/insert_letter runs/t5_a --steps 60000 --goal --crop --grip-mode reg
link_weights t5_a insert_letter

# v2 + r1 -> t5_b
for k in 0 1 2; do
  python dev/gen_demos.py insert_letter data/demos/insert_letter_v2 $((5100000 + k * 200)) 200 --img-stride 2 --crop grip --set V2=true
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_v2 $((5300000 + k * 100)) 100 --student runs/t5_a/weights \
    --set V2=true --crop grip --img-stride 2
done
python dev/train_act.py insert_letter data/demos/insert_letter_v2 runs/t5_b --steps 60000 --goal --crop --grip-mode reg
link_weights t5_b insert_letter

# r2 -> t5_c
R2="--set V2=true --set DESCENT_NOISE=0.3 --set MISALIGN_P=0.25 --crop grip --img-stride 2"
for k in 0 1 2; do
  python dev/gen_demos.py insert_letter data/demos/insert_letter_r2 $((5600000 + k * 70)) 70 $R2
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r2 $((5310000 + k * 100)) 100 --student runs/t5_b/weights $R2
done
V2D=data/demos/insert_letter_v2
link_data insert_letter_c $V2D/*_51000??.npz $V2D/*_51001??.npz $V2D/*_51002[0-4]?.npz $V2D/*_53*.npz data/demos/insert_letter_r2/*.npz
python dev/train_act.py insert_letter data/demos/insert_letter_c runs/t5_c --steps 80000 --goal --crop --grip-mode reg
link_weights t5_c insert_letter

# r3 -> t5_d, t5_d2
for k in 0 1 2; do
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r3 $((5320000 + k * 100)) 100 --student runs/t5_c/weights $R2
done
link_data insert_letter_d $V2D/*_51000??.npz $V2D/*_51001??.npz $V2D/*_51002[0-4]?.npz data/demos/insert_letter_r2/*.npz \
  data/demos/insert_letter_r3/*.npz
python dev/train_act.py insert_letter data/demos/insert_letter_d runs/t5_d --steps 80000 --goal --crop --grip-mode reg
python dev/train_act.py insert_letter data/demos/insert_letter_d runs/t5_d2 --steps 80000 --goal --crop --grip-mode reg
link_weights t5_d insert_letter; link_weights t5_d2 insert_letter

# v3 + r5 -> t5_f
V3="--set V2=true --set DESCENT_NOISE=0.3 --set MISALIGN_P=0.25 --set QUICK_RELEASE=true --set SETTLE_CAP=3 --set RESUME_NEAR=0.06 --crop grip --img-stride 2"
for k in 0 1 2 3; do
  python dev/gen_demos.py insert_letter data/demos/insert_letter_v3 $((5920000 + k * 125)) 125 $V3
done
for k in 0 1; do
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r5 $((5930000 + k * 125)) 125 --student runs/t5_d/weights $V3 \
    --triggers low=0.25,stall=0.4,carry=0.15,early=0.2
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r5 $((5940000 + k * 125)) 125 --student runs/t5_d2/weights $V3 \
    --triggers low=0.25,stall=0.4,carry=0.15,early=0.2
done
link_data insert_letter_f data/demos/insert_letter_v3/*.npz data/demos/insert_letter_r5/*.npz
python dev/train_act.py insert_letter data/demos/insert_letter_f runs/t5_f --steps 80000 --goal --crop --grip-mode reg
link_weights t5_f insert_letter

# r6 -> t5_g
for k in $(seq 0 7); do
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r6 $((5950000 + k * 100)) 100 --student runs/t5_f/weights $V3 \
    --set NEAR_NO_RISE=true --triggers stall=0.5,low=0.2,early=0.15,carry=0.15
done
python - > data/demos/r5_keep.txt <<'EOF'   # r5 without the "near" resumes whose TCP rose > 3 mm
import glob, json, sys
import numpy as np
sys.path.insert(0, ".")
from mfagent import robot
for path in sorted(glob.glob("data/demos/insert_letter_r5/*.npz")):
    z = np.load(path, allow_pickle=True)
    if json.loads(str(z["meta"])).get("resume") == "near":
        tz = [robot.tcp_pose(x[:7].astype(float))[2, 3] for x in z["qpos"][:12]]
        if max(tz) - tz[0] > 0.003:
            continue
    print(path)
EOF
link_data insert_letter_g data/demos/insert_letter_v3/*.npz data/demos/insert_letter_r6/*.npz $(cat data/demos/r5_keep.txt)
python dev/train_act.py insert_letter data/demos/insert_letter_g runs/t5_g --steps 80000 --goal --crop --grip-mode reg
link_weights t5_g insert_letter

# r7 -> t5_h (fine-tune)
for k in $(seq 0 15); do
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r7 $((6100000 + k * 50)) 50 --student runs/t5_g/weights $V3 \
    --set NEAR_NO_RISE=true --set HOME_Z=0.001 --triggers stall=0.25,seat=0.3,low=0.15,home=0.1,early=0.1,carry=0.1
done
link_data insert_letter_h data/demos/insert_letter_g/*.npz data/demos/insert_letter_r7/*.npz
python dev/train_act.py insert_letter data/demos/insert_letter_h runs/t5_h --steps 30000 --goal --crop --grip-mode reg \
  --init runs/t5_g/last.pt --lr 5e-5 --lr-backbone 5e-6
link_weights t5_h insert_letter

# r8 -> t5_j (fine-tune)
for k in $(seq 0 15); do
  python dev/gen_takeover.py insert_letter data/demos/insert_letter_r8 $((6101000 + k * 50)) 50 --student runs/t5_h/weights $V3 \
    --set NEAR_NO_RISE=true --set HOME_Z=0.001 --triggers stall=0.25,seat=0.3,low=0.15,early=0.15,carry=0.15
done
link_data insert_letter_j data/demos/insert_letter_h/*.npz data/demos/insert_letter_r8/*.npz
python dev/train_act.py insert_letter data/demos/insert_letter_j runs/t5_j --steps 30000 --goal --crop --grip-mode reg \
  --init runs/t5_h/last.pt --lr 5e-5 --lr-backbone 5e-6
```

(Our r8 run used 12 jobs of 50 seeds from 6,101,000 and 40 jobs of 5 seeds from 6,101,600, which covers the same
seeds as the loop above.)

## Building and checking a submission folder

```bash
python dev/build_lsubmission.py --out hf_submission_l --team NAME --email ADDR --text runs/text_clf_v1 \
  --weights grasp_part=runs/t1_g/last.pt --weights grasp_move_mug=runs/t2_c/last.pt \
  --weights toggle_switch_table=runs/t3_d/last.pt --weights put_blocks_into_boxes=runs/t4_a/last.pt \
  --weights insert_letter=runs/t5_j/last.pt
echo '{"seeds": [2640001, 2640002, 2640003]}' > $PWD/seeds.json
SUBMISSION=$PWD/hf_submission_l dev/http_eval.sh eval.eval_insert_letter $PWD/seeds.json $PWD/runs/http_t5 3
```

The builder copies `lpolicy/`, the checkpoints and the classifier into the official submission template and runs
its self-test. `dev/http_eval.sh` serves the folder and runs one official evaluation script against it.
