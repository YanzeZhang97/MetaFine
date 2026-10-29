"""Task routines keyed by the harness ``task_id``."""
from __future__ import annotations

from .. import motion


def _idle(ctx):
    yield from motion.hold(ctx)


def preload() -> None:
    """Import every routine and load the bundled assets now, so a missing module, dependency
    or file fails at server start (where the build check sees it) instead of silently
    turning a whole task into holding still."""
    from . import grasp_move_mug, grasp_part, insert_letter, put_blocks, toggle_switch  # noqa: F401
    from ..perception import pixelmlp, selfmask
    pixelmlp._load()
    selfmask._load()


def get_routine(task_id: str):
    if task_id == "put_blocks_into_boxes":
        from .put_blocks import routine
        return routine
    if task_id == "toggle_switch_table":
        from .toggle_switch import routine
        return routine
    if task_id == "grasp_part":
        from .grasp_part import routine
        return routine
    if task_id == "grasp_move_mug":
        from .grasp_move_mug import routine
        return routine
    if task_id == "insert_letter":
        from .insert_letter import routine
        return routine
    return _idle
