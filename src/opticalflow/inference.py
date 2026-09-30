import numpy as np

from opticalflow.preprocessing import prepare


def infer(model, first, second, intensity_max):
    if first.shape != second.shape:
        raise ValueError("Frame pair dimensions differ")
    height, width = first.shape[:2]
    flow = model.predict(prepare(first, intensity_max), prepare(second, intensity_max))
    flow = flow[:, :height, :width].permute(1, 2, 0).numpy().astype(np.float32)
    if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
        raise ValueError("Nonfinite or malformed flow output")
    return np.ascontiguousarray(flow)


def statistics(flow):
    values = flow.astype(np.float64)
    magnitude = np.linalg.norm(values, axis=2)
    return {
        "pixels": int(magnitude.size),
        "mean_u": float(values[..., 0].mean()),
        "mean_v": float(values[..., 1].mean()),
        "mean_magnitude": float(magnitude.mean()),
        "std_magnitude": float(magnitude.std()),
        "median_magnitude": float(np.median(magnitude)),
        "p95_magnitude": float(np.percentile(magnitude, 95)),
        "max_magnitude": float(magnitude.max()),
    }
