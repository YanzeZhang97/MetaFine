"""Dev-only helpers: build competition envs exactly like the official eval scripts.

Never shipped in the submission. Usage: from dev.simenv import make_task_env
"""
import os
import sys

# The official competition repository (https://github.com/Hiangx-robotics/MetaFine-Challenge_CoRL): set METAFINE_REPO,
# or clone it into this folder.
REPO = os.environ.get("METAFINE_REPO") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                       "MetaFine-Challenge_CoRL")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# MF_RENDER_BACKEND=cpu: render the cameras on the CPU (Mesa llvmpipe) instead of a GPU, for when the GPUs
# are full. Physics is physx_cpu either way. The hand images match the GPU renders to within ~0.2 grey
# levels on average (<0.1 % of pixels, at silhouette edges, differ by more than 8). Run with
# CUDA_VISIBLE_DEVICES="" so torch and SAPIEN stay off the GPUs. The Vulkan driver must be llvmpipe's
# (with the NVIDIA one SAPIEN segfaults): forced here, before SAPIEN loads (the login shell exports the
# NVIDIA ICD; an already-imported SAPIEN keeps its driver).
RENDER_BACKEND = os.environ.get("MF_RENDER_BACKEND") or None
if RENDER_BACKEND == "cpu":
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"

import numpy as np

TASK_CFG = {
    "grasp_part": "eval/configs/grasp_part.yaml",
    "grasp_move_mug": "eval/configs/grasp_move_mug.yaml",
    "toggle_switch_table": "eval/configs/toggle_switch_table.yaml",
    "put_blocks_into_boxes": "eval/configs/put_blocks_into_boxes.yaml",
    "insert_letter": "eval/configs/insert_letter.yaml",
}


def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    x = np.asarray(x)
    return x[0] if x.ndim > 1 and x.shape[0] == 1 else x


def load_cfg(task):
    """Task config as the eval scripts load it. MF_SENSOR_WH=WxH (e.g. 224x224) overrides the
    YAML sensor size, which applies to every camera, the hand camera included."""
    from utils.eval_common import load_task_config
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        cfg = load_task_config(TASK_CFG[task])
    finally:
        os.chdir(cwd)
    wh = os.environ.get("MF_SENSOR_WH")
    if wh:
        w, h = (int(v) for v in wh.lower().split("x"))
        cfg["sensor"] = {**(cfg.get("sensor") or {}), "width": w, "height": h}
    return cfg


def make_task_env(task, variant=None, profile="clean", extra=None):
    """Env for (task, variant) under one of the 7 perception profiles; returns (env, cfg, variant)."""
    from utils.eval_common import build_base_env_kwargs, build_perception_profiles
    cfg = load_cfg(task)
    variant = variant or cfg["perception"]["default_variant"]
    prof = build_perception_profiles(build_base_env_kwargs(cfg), cfg)[profile]
    pert = {k: v for k, v in prof.items() if k.startswith("eval_")}
    extra = {**pert, **(extra or {})}
    if RENDER_BACKEND:
        extra.setdefault("render_backend", RENDER_BACKEND)
    if os.environ.get("MF_SMALL_BASE_CAM") == "1":
        # The policy reads the hand camera only: render the base camera at 16 x 16 (about halves CPU rendering).
        # The hand camera, the scene and the RNG draws are unchanged.
        from utils.eval_common import sensor_configs_from_task
        sc = sensor_configs_from_task(cfg)
        sc["base_camera"] = {**sc.get("base_camera", {}), "width": 16, "height": 16}
        extra.setdefault("sensor_configs", sc)
    cwd = os.getcwd()
    os.chdir(REPO)  # asset paths in the official code are repo-relative in places
    try:
        if task == "grasp_part":
            from eval.eval_grasp_part import _make_env
            env = _make_env(build_base_env_kwargs(cfg, extra={"part_name": variant, **extra}))
        elif task == "grasp_move_mug":
            from eval.eval_grasp_move_mug import _make_env
            env = _make_env(cfg, variant, extra=extra)
        elif task == "toggle_switch_table":
            from eval.eval_toggle_switch import _make_env, _variant_extra
            env = _make_env(build_base_env_kwargs(cfg, extra={**_variant_extra(cfg, variant), **extra}))
        elif task == "put_blocks_into_boxes":
            from eval.eval_put_blocks import _make_env
            env = _make_env(build_base_env_kwargs(cfg, extra={"special_cube": variant, **extra}))
        elif task == "insert_letter":
            from eval.eval_insert_letter import _make_env
            env = _make_env(build_base_env_kwargs(cfg, extra={"target_letter": variant, **extra}))
        else:
            raise KeyError(task)
    finally:
        os.chdir(cwd)
    return env, cfg, variant


def obs_to_policy_inputs(obs):
    """What the HTTP harness sends: 9-dim qpos and uint8 RGB per camera."""
    state = to_np(obs["agent"]["qpos"]).astype(np.float32).reshape(-1)
    images = {cam: to_np(d["rgb"]).astype(np.uint8) for cam, d in obs["sensor_data"].items()}
    return state, images
