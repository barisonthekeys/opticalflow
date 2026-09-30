"""Fixed intensity normalization and replicate padding; never resize."""

import numpy as np
import torch
import torch.nn.functional as F


def prepare(array, intensity_max=None):
    if array.dtype == np.uint8:
        scale = 255.0 if intensity_max is None else float(intensity_max)
    elif array.dtype == np.uint16 and intensity_max is not None:
        scale = float(intensity_max)
    else:
        raise ValueError(
            "uint16 requires an explicit intensity_max for the camera bit depth"
        )
    if not np.isfinite(scale) or scale <= 0 or array.max() > scale:
        raise ValueError("Invalid intensity_max or input exceeds configured maximum")
    tensor = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1)[None] / scale
    height, width = array.shape[:2]
    padded_h = max(128, (height + 7) // 8 * 8)
    padded_w = max(128, (width + 7) // 8 * 8)
    return F.pad(tensor, (0, padded_w - width, 0, padded_h - height), mode="replicate")
