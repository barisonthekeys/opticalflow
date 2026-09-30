#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opticalflow.data import build_manifest
from opticalflow.experiment import write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, build_manifest(args.data_root, ["LF", "MF", "HF"]))
