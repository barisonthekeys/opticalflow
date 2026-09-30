"""Strict numbered-frame discovery, content fingerprints, and image decoding."""

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def decode(payload, suffix):
    if suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(io.BytesIO(payload))
    else:
        with Image.open(io.BytesIO(payload)) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame image containers are not supported")
            array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected grayscale or RGB image, got {array.shape}")
    if array.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected uint8/uint16, got {array.dtype}")
    return array


def build_manifest(root, sequences):
    root = Path(root).resolve()
    result = {"schema_version": 1, "data_root": str(root), "sequences": {}}
    for sequence in sequences:
        directory = root / sequence
        files = [p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS]
        indexed = []
        for path in files:
            match = re.search(r"(\d+)$", path.stem)
            if not match:
                raise ValueError(
                    f"Frame filename must end with its frame number: {path}"
                )
            indexed.append((int(match[1]), path))
        indexed.sort()
        numbers = [number for number, _ in indexed]
        if len(numbers) < 2:
            raise ValueError(f"{sequence}: need at least two frames")
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"{sequence}: duplicate or missing frame numbers")
        rows = []
        geometry = None
        for number, path in indexed:
            payload = path.read_bytes()
            array = decode(payload, path.suffix)
            signature = (array.shape, str(array.dtype))
            if geometry is not None and geometry != signature:
                raise ValueError(f"Geometry/dtype changes within {sequence}: {path}")
            geometry = signature
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "frame": number,
                    "sha256": digest(payload),
                    "bytes": len(payload),
                    "height": array.shape[0],
                    "width": array.shape[1],
                    "dtype": str(array.dtype),
                }
            )
        result["sequences"][sequence] = rows
    return result


def load_frame(root, row):
    path = Path(root) / row["path"]
    payload = path.read_bytes()
    if digest(payload) != row["sha256"]:
        raise ValueError(f"Input changed since manifest: {path}")
    return decode(payload, path.suffix)
