"""Append-only experiment execution with per-run provenance and integrity records."""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torchvision
import yaml

from opticalflow.data import build_manifest, digest, load_frame
from opticalflow.inference import infer, statistics
from opticalflow.models.raft import RaftBaseline, choose_device
from opticalflow.visualization import render


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def command(args, root):
    result = subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def validate(config):
    if config["device"] not in ("mps", "cuda"):
        raise ValueError(
            "Experiment 001 requires device: mps or cuda; CPU inference is disabled"
        )
    if (
        config["model"] != "raft_large"
        or config["weights"] != "C_T_SKHT_V2"
        or config["stride"] != 1
    ):
        raise ValueError("EXP001 requires raft_large, C_T_SKHT_V2, stride=1")
    if config["num_flow_updates"] < 1 or config["cpu_threads"] < 1:
        raise ValueError("updates and cpu_threads must be positive")
    if (
        not np.isfinite(config["visualization_max_px"])
        or config["visualization_max_px"] <= 0
    ):
        raise ValueError("visualization_max_px must be positive and finite")
    if not config["sequences"] or len(set(config["sequences"])) != len(
        config["sequences"]
    ):
        raise ValueError("sequences must be nonempty and unique")
    if any(s not in ("LF", "MF", "HF") for s in config["sequences"]):
        raise ValueError("Unknown sequence")
    if config["experiment"] != "EXP001_raft_baseline":
        raise ValueError("Unexpected experiment identifier")


def capture(root, output):
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    snapshot = output / "source"
    snapshot.mkdir()
    for name in ("src", "scripts", "configs", "tests", "environment"):
        source = root / name
        if source.exists():
            shutil.copytree(
                source,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    for name in (
        "pyproject.toml",
        "README.md",
        "EXPERIMENT001.md",
        "TRANSFER.md",
        ".gitignore",
    ):
        if (root / name).exists():
            shutil.copy2(root / name, snapshot / name)
    write_json(
        output / "git.json",
        {
            key: command(["git", *args], root)
            for key, args in {
                "head": ["rev-parse", "HEAD"],
                "status": ["status", "--porcelain=v1"],
                "diff": ["diff", "--binary", "HEAD"],
                "untracked": ["ls-files", "--others", "--exclude-standard"],
            }.items()
        },
    )
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK"
            ),
            "pip_freeze": command([sys.executable, "-m", "pip", "freeze"], root),
            "conda_explicit": command(
                [conda, "list", "-p", sys.prefix, "--explicit"],
                root,
            )
            if conda
            else {"unavailable": True},
            "argv": sys.argv,
            "reproducibility_note": "Seeds and exact inputs recorded; bitwise equality across devices is not guaranteed.",
        },
    )


def run(config, root, limit=None, data_root=None):
    validate(config)
    root = Path(root).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    input_root = (
        Path(data_root).resolve()
        if data_root
        else (root / config["data_root"]).resolve()
    )
    manifest = build_manifest(input_root, config["sequences"])
    selected = {
        seq: list(pairwise(rows))[:limit] for seq, rows in manifest["sequences"].items()
    }
    raw_bytes = sum(
        a["height"] * a["width"] * 8 + 128
        for pairs in selected.values()
        for a, _ in pairs
    )
    run_root = root / config["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_root).free < raw_bytes * 1.4 + 100_000_000:
        raise OSError(f"Insufficient storage: raw arrays alone need {raw_bytes} bytes")
    identity = f"{config['experiment']}_{'smoke' if limit else 'full'}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
    output = run_root / identity
    output.mkdir(exist_ok=False)
    print(f"RUN {output}", flush=True)
    status = {
        "started_utc": now(),
        "status": "running",
        "completed_pairs": 0,
        "planned_pairs": sum(map(len, selected.values())),
        "raw_bytes_estimate": raw_bytes,
        "limit_per_sequence": limit,
        "flow_convention": "forward first->second; u right, v down; pixels/frame interval",
        "padding": "replicate right/bottom to multiples of 8, minimum 128; crop output to original size",
        "grayscale": "replicated into RGB; fixed intensity scale then weights.transforms()",
        "visualization": "HSV hue=atan2(v,u)/(2*pi); value=clip(magnitude/visualization_max_px); black=zero",
        "metrics_note": "Descriptive flow statistics only; no ground truth accuracy or physical velocity implied.",
    }
    summary = {}
    try:
        write_json(output / "config.json", config)
        write_json(output / "manifest.json", manifest)
        capture(root, output)
        model = RaftBaseline(config, root / config["cache_root"])
        write_json(output / "model.json", model.metadata)
        with (output / "pairs.csv").open("x", newline="") as handle:
            writer = None
            for sequence, pairs in selected.items():
                destination = output / sequence
                destination.mkdir()
                aggregate = {
                    "pairs": 0,
                    "pixels": 0,
                    "sum_magnitude": 0.0,
                    "sum_square_magnitude": 0.0,
                    "sum_u": 0.0,
                    "sum_v": 0.0,
                    "max_magnitude": 0.0,
                }
                for a, b in pairs:
                    start = time.perf_counter()
                    first, second = load_frame(input_root, a), load_frame(input_root, b)
                    flow = infer(model, first, second, config["intensity_max"])
                    stem = f"{a['frame']:06d}_{b['frame']:06d}"
                    with (destination / f"{stem}.npy").open("xb") as array_file:
                        np.save(array_file, flow, allow_pickle=False)
                    render(
                        flow,
                        destination / f"{stem}.png",
                        config["visualization_max_px"],
                    )
                    stats = statistics(flow)
                    row = {
                        "sequence": sequence,
                        "first": a["path"],
                        "second": b["path"],
                        "flow": f"{sequence}/{stem}.npy",
                        "seconds": time.perf_counter() - start,
                        **stats,
                    }
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    n = stats["pixels"]
                    aggregate["pairs"] += 1
                    aggregate["pixels"] += n
                    for field in ("magnitude", "u", "v"):
                        aggregate[f"sum_{field}"] += n * stats[f"mean_{field}"]
                    aggregate["sum_square_magnitude"] += n * (
                        stats["std_magnitude"] ** 2 + stats["mean_magnitude"] ** 2
                    )
                    aggregate["max_magnitude"] = max(
                        aggregate["max_magnitude"], stats["max_magnitude"]
                    )
                    status["completed_pairs"] += 1
                    print(
                        f"{sequence} {stem} {status['completed_pairs']}/{status['planned_pairs']} {row['seconds']:.2f}s",
                        flush=True,
                    )
                n = aggregate["pixels"]
                mean = aggregate["sum_magnitude"] / n
                summary[sequence] = {
                    "pairs": aggregate["pairs"],
                    "pixels": n,
                    "mean_magnitude": mean,
                    "std_magnitude": max(
                        0.0, aggregate["sum_square_magnitude"] / n - mean**2
                    )
                    ** 0.5,
                    "mean_u": aggregate["sum_u"] / n,
                    "mean_v": aggregate["sum_v"] / n,
                    "max_magnitude": aggregate["max_magnitude"],
                }
        write_json(output / "summary.json", summary)
        status["status"] = "completed"
    except BaseException:
        status["status"] = "failed"
        status["error"] = traceback.format_exc()
        raise
    finally:
        status["finished_utc"] = now()
        write_json(output / "status.json", status)
        checksums = {
            p.relative_to(output).as_posix(): digest(p.read_bytes())
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(output / "checksums.json", checksums)
        # Sealed outputs are never reopened by the runner; new invocation = new run.
        for path in output.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_dir():
                path.chmod(0o555)
        output.chmod(0o555)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--config", default="configs/raft_baseline.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        help="First N consecutive pairs per sequence; omit for full run",
    )
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    config_path = args.root / args.config
    config = yaml.safe_load(config_path.read_text())
    validate(config)
    choose_device(config["device"])
    run(config, args.root, args.limit, args.data_root)
