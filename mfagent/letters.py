"""T5 geometry copied from core/letter_glyphs.py and core/env.py (InsertLetterEnv).

Letter pegs are unions of axis-aligned boxes on a 2 mm grid, 36 mm tall, with the
actor origin at the bottom centre of the glyph frame (x right, y toward the board).
The board's through-holes inflate every stroke by 4 mm per side; slot centres
are fixed and the goal yaw is 0 (o is 180-degree symmetric).
"""
from __future__ import annotations

import numpy as np

U = 0.002
PEG_H = 0.036
BOARD_H = 0.014
BOARD_Y = 0.14
POCKET_MARGIN = 0.004
SLOT_X = {"C": -0.084, "o": -0.028, "R": 0.028, "L": 0.084}
SYMMETRY = {"C": 2 * np.pi, "o": np.pi, "R": 2 * np.pi, "L": 2 * np.pi}  # yaw period

# RGB of each peg (core/letter_glyphs.py PEG_COLORS).
PEG_RGB = {"C": (0.90, 0.18, 0.18), "o": (0.18, 0.40, 0.90), "R": (0.95, 0.78, 0.15), "L": (0.18, 0.72, 0.28)}

# Strokes (x0, y0, x1, y1) in grid units, glyph frame.
GLYPHS = {
    "C": [(-10, 8, 10, 13), (-10, -8, -5, 8), (-10, -13, 10, -8)],
    "o": [(-8, -2, 8, 3), (-8, -13, 8, -8), (-8, -8, -3, -2), (3, -8, 8, -2)],
    "R": [(-10, -13, -5, 13), (-5, 8, 10, 13), (5, 0, 10, 8), (-5, -1, 10, 4), (0, -7, 6, -1), (4, -13, 10, -7)],
    "L": [(-10, -13, -5, 13), (-5, -13, 10, -8)],
}

# Squeeze grasps. "x": fingers close along the glyph x axis on the row at glyph y = c;
# "y": fingers close along the glyph y axis on the column at glyph x = c. The squeeze
# centres whatever the letter spans on that line between the fingers.
GRASPS = {
    "C": [("x", 0.0), ("y", -0.015)],
    "o": [("x", -0.010), ("y", 0.0)],
    "R": [("x", 0.012), ("y", -0.015)],
    "L": [("x", 0.010), ("y", -0.015)],
}
# TCP height while holding the peg by its upper part (core/letter_glyphs.py GRASP_Z).
GRASP_TCP_Z = 0.028


def strokes_m(letter: str) -> np.ndarray:
    """(N, 4) stroke rectangles in metres."""
    return np.asarray(GLYPHS[letter], dtype=float) * U


def line_span(letter: str, axis: str, c: float) -> tuple[float, float]:
    """Extent of the letter along the grasp line (glyph x for axis "x" at y = c, glyph y for "y" at x = c)."""
    out = []
    for x0, y0, x1, y1 in strokes_m(letter):
        if axis == "x" and y0 - 1e-9 <= c <= y1 + 1e-9:
            out.append((x0, x1))
        if axis == "y" and x0 - 1e-9 <= c <= x1 + 1e-9:
            out.append((y0, y1))
    return min(a for a, _ in out), max(b for _, b in out)


def grasp_point(letter: str, axis: str = "x", c: float | None = None) -> np.ndarray:
    """Glyph-frame (x, y) the TCP sits over so the squeeze centres the line's span."""
    if c is None:
        axis, c = GRASPS[letter][0]
    lo, hi = line_span(letter, axis, c)
    mid = 0.5 * (lo + hi)
    return np.array([mid, c]) if axis == "x" else np.array([c, mid])


def grasp_width(letter: str, axis: str, c: float) -> float:
    lo, hi = line_span(letter, axis, c)
    return hi - lo


def slot_pose(letter: str) -> np.ndarray:
    """World (x, y, yaw) of the target pose of the peg origin."""
    return np.array([SLOT_X[letter], BOARD_Y, 0.0])
