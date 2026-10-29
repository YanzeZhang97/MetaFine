"""Image preprocessing shared by data generation and the runtime (they must match exactly)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMG = 224  # network input side


def square(img) -> np.ndarray:
    """Centre square of a non-square image. The hand camera's vertical field of view is fixed, so for
    e.g. 640 x 480 the centre 480 x 480 sees what the 512 x 512 training images saw."""
    h, w = img.shape[:2]
    if h == w:
        return img
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def resize_rgb(img, size: int = IMG) -> np.ndarray:
    """HxWx3 uint8 -> size x size x 3 uint8 (centre square, then bilinear with antialiasing, on CPU)."""
    img = square(img)
    t = torch.as_tensor(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
    if t.shape[-2:] != (size, size):
        t = F.interpolate(t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return t[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def center_crop(img, size: int = IMG, ref: int = 512) -> np.ndarray:
    """Centre crop covering size/ref of the image side (224 of 512 px: native resolution at the
    eval's 512 x 512), resized to size x size when the camera resolution differs."""
    img = square(img)
    n = img.shape[0]
    c = max(1, round(n * size / ref))
    y0 = (n - c) // 2
    return resize_rgb(np.ascontiguousarray(img[y0:y0 + c, y0:y0 + c]), size)


# Second-stream crops: (centre x, centre y, side), as fractions of the image square. "grip": the
# region below and right of centre where the hand camera sees a held T5 peg and the board under it
# (266 of 512 px, so about twice the main stream's resolution).
CROP_BOXES = {"grip": (0.70, 0.74, 0.52)}


def crop(img, kind: str = "center", size: int = IMG) -> np.ndarray:
    """The named second-stream crop of a hand image, size x size."""
    if kind == "center":
        return center_crop(img, size)
    cx, cy, side = CROP_BOXES[kind]
    img = square(img)
    n = img.shape[0]
    c = max(1, round(n * side))
    x0 = min(max(round(n * cx - c / 2), 0), n - c)
    y0 = min(max(round(n * cy - c / 2), 0), n - c)
    return resize_rgb(np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c]), size)
