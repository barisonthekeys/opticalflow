"""HSV: hue=direction (right=red, down=yellow-green); value=fixed-scale magnitude."""

import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image


def render(flow, destination, maximum):
    magnitude = np.linalg.norm(flow, axis=2)
    hue = (np.arctan2(flow[..., 1], flow[..., 0]) / (2 * np.pi)) % 1
    hsv = np.stack((hue, np.ones_like(hue), np.clip(magnitude / maximum, 0, 1)), axis=2)
    Image.fromarray(np.rint(hsv_to_rgb(hsv) * 255).astype(np.uint8)).save(destination)
