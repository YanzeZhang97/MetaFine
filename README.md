# Learning from an Imitable Teacher: Solution to the MetaFine Challenge

Code for Team Friday's entry to the [MetaFine Open Competition](https://robofinemani2026.github.io/competition.html)
(RoboFineMani workshop, CoRL 2026), which placed **second** with an overall score of **0.947** on five fine-grained,
language-conditioned manipulation tasks. The organizers' π0 baseline scores 0.430.

Technical report: *Learning from an Imitable Teacher: Solution to the MetaFine Challenge* (arXiv link to be added).

## Method in brief

- **Instruction classifier.** A fine-tuned all-MiniLM-L6-v2 sentence encoder (22M parameters) with one linear head
  per task maps the instruction to a task variant.
- **Visuomotor policy.** One ACT-style transformer per task (19.4M parameters, no CVAE): an ImageNet ResNet-18 on
  the 224×224 hand-camera image with FiLM conditioning on the variant, a 4+4-layer transformer over the image
  tokens, the joint positions and the variant, and chunks of 32 joint-delta actions, of which 8 are executed. T5
  adds a second stream that sees a crop around the gripper.
- **Data.** A privileged teacher runs scripted routines (motion primitives with inverse kinematics) with perception
  answered from the simulator state. HG-DAgger-style takeovers record its corrections from the student's own
  mistakes. Most of the work went into making the teacher *imitable* by a memoryless student: no long still phases,
  randomized habits, staged and recovered errors, takeovers consistent with the plain demonstrations, and a gentle
  release.
- At test time only the two networks run, on the allowed observations (hand camera, joint positions, instruction).
  The teacher and the simulator state are used only to generate training data.

## Official results

| Task | P | U | B | Score | π0 baseline |
|---|---|---|---|---|---|
| T1 `grasp_part` | 0.865 | 0.900 | 0.900 | 0.888 | 0.332 |
| T2 `grasp_move_mug` | 0.983 | 0.983 | 0.983 | 0.983 | 0.931 |
| T3 `toggle_switch_table` | 1.000 | 1.000 | 1.000 | 1.000 | 0.515 |
| T4 `put_blocks_into_boxes` | 0.995 | 0.983 | 0.983 | 0.987 | 0.331 |
| T5 `insert_letter` | 0.880 | 0.875 | 0.875 | 0.877 | 0.042 |
| **Overall** | | | | **0.947** | 0.430 |

P: Perception (area under the success curve over camera and lighting perturbations), U: Understanding, B: Behavior.

## Repository layout

```
lpolicy/      the submitted runtime policy: model, image preprocessing, instruction classifier, reset/act loop
dev/          simulator wrapper, in-process copy of the official evaluation loop, privileged teacher,
              data generation (plain demonstrations and takeovers), training, submission builder
mfagent/      scripted routines (motion primitives, inverse kinematics) that the teacher runs; not used at test time
docs/         REPRODUCE.md: exact commands and seeds for every shipped network
```

## Setup

```bash
conda create -n metafine python=3.10 -y && conda activate metafine
git clone https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL.git   # official competition code and assets
(cd MetaFine-Challenge_CoRL && pip install -e .)
pip install -r requirements.txt
export METAFINE_REPO=$PWD/MetaFine-Challenge_CoRL   # optional when the clone sits in this folder

# pretrained initializations (only needed for training)
python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='data/models/all-MiniLM-L6-v2')"
```

To render on the CPU instead of a GPU, set `MF_RENDER_BACKEND=cpu CUDA_VISIBLE_DEVICES=""` (see `dev/simenv.py`).

## Evaluating the submitted networks

The weights of our final submission (five control networks and the instruction classifier, 443 MB) are attached
to the [GitHub release](https://github.com/YanzeZhang97/MetaFine/releases) as `MetaFine-weights.tar.gz`
(SHA-256 `f649a3bd415ce07778a83fa1f000f65c48a0d7b716aee74485959c1330b058c5`).

```bash
tar xzf MetaFine-weights.tar.gz -C lpolicy/      # -> lpolicy/weights/<task>.pt and lpolicy/weights/text/
mkdir -p runs/eval

# one episode in the in-process official loop
POLICY=lpolicy python dev/official_loop.py toggle_switch_table red 2900001

# a task: every variant on clean episodes, plus the default variant under three lighting levels
dev/eval_lpolicy.sh lpolicy/weights insert_letter runs/eval/t5.jsonl 20 "light_l1 light_l2 light_l3" 2900001
```

`dev/build_lsubmission.py` packs the networks into the official submission template, and `dev/http_eval.sh` runs
an official evaluation script against the packed folder over HTTP (see `docs/REPRODUCE.md`).

## Training

`docs/REPRODUCE.md` lists every data-generation and training command, with seeds, for the instruction classifier
and the five networks. The main entry points:

```bash
python dev/gen_demos.py TASK OUT_DIR FIRST_SEED N_SEEDS [--set TEACHER_OPTION=value ...]      # plain demonstrations
python dev/gen_takeover.py TASK OUT_DIR FIRST_SEED N_SEEDS --student runs/RUN/weights [...]    # takeovers
python dev/train_act.py TASK DATA_DIR RUN_DIR --steps 40000 --goal                             # one network
```

Teacher options (`--set`) are documented in `dev/teacher.py`, next to where each is defined.

## Citation

```bibtex
@article{zhang2026imitable,
  title   = {Learning from an Imitable Teacher: Solution to the MetaFine Challenge},
  author  = {Zhang, Yanze and Liu, Hao and Hao, Haochang and Shen, Zhixiang and Miao, Chunyu and Yang, Yuyao},
  journal = {arXiv preprint},
  year    = {2026}
}

@article{xu2026beyond,
  title   = {Beyond Binary Success: A Diagnostic Meta-Evaluation Framework for Fine-Grained Manipulation},
  author  = {Xu, He-Yang and Zhang, Pengyuan and Ge, Zongyuan and Hao, Xiaoshuai and Belongie, Serge and Geng, Xin
             and Peng, Yuxin and Wei, Xiu-Shen},
  journal = {arXiv preprint arXiv:2605.19986},
  year    = {2026}
}
```

## Acknowledgments

We thank the MetaFine organizers for the competition, its simulator tasks and evaluation code. The policy follows
ACT (Zhao et al., RSS 2023); the instruction classifier is fine-tuned from
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache 2.0).

## License

MIT (see `LICENSE`).
