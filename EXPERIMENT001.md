# Experiment 001 — pretrained RAFT-Large baseline

Use Conda environment `of`. No package installation is needed in the inspected environment.

```sh
conda activate of
cd /Users/baristura/Projects/opticalflow
python -m pytest -q
python scripts/run_experiment.py --limit 2
# Inspect the smoke run status, arrays, PNGs, timing and summary before the full run:
python scripts/run_experiment.py
```

The first inference downloads torchvision's explicit `Raft_Large_Weights.C_T_SKHT_V2`
checkpoint into ignored `.cache/torch/checkpoints/`. `device: mps` is mandatory. The command checks MPS before scanning data and
fails immediately if unavailable. CPU inference and CPU operator fallback are disabled. There is no silent retry or device change after inference starts.

## Scientific definition

Each sequence is one acquisition, with filenames ending in contiguous integer frame
numbers. Numeric sorting avoids lexical ordering errors; gaps and duplicate numbers
are rejected. All consecutive pairs use stride 1, batch size 1, 12 RAFT updates,
eval/inference mode, float32, no adaptation, no resizing. Grayscale is replicated
into RGB. uint8 values are divided by 255; uint16 requires the camera's explicit
`intensity_max` (e.g. 4095 for valid 12-bit data). Per-frame min/max normalization
is never used. The weight transform maps [0,1] into [-1,1]. Right/bottom replicate
padding reaches multiples of eight and the model minimum 128; output padding is
cropped off without vector scaling.

A `.npy` contains H×W×2 float32 forward displacement from the first image to the
second: channel 0 is horizontal/right-positive u, channel 1 vertical/down-positive v.
Units are pixels per frame interval. Frame rate and spatial calibration are not yet
specified, so these are not physical velocities. Pretrained flow on boiling images
is an exploratory baseline, not validated ground-truth motion.

PNGs use HSV direction hue and magnitude brightness with one common fixed maximum
(default 20 px); values above it saturate visually but raw arrays remain unchanged.
Right is red, down yellow-green, left cyan, up purple; zero is black. `pairs.csv`
contains per-pair means, standard deviation, median, p95 and max magnitude plus u/v
means. `summary.json` aggregates means and population standard deviation over all
pixels, not averages of percentiles. Timings include decoding and output writing.

## Bookkeeping

Every invocation creates a unique UTC/UUID run, never overwrites or resumes an old
run, and records exact config, full input manifest (SHA-256, shape, dtype, frame
number), selected pair paths, code/config/test/environment snapshots, Git HEAD,
dirty diff/status/untracked names, package freeze, Conda explicit package list,
platform, device availability, checkpoint URL/hash, seed and timestamps. Each
frame is rehashed before decoding at inference to detect changed inputs. Runs
record completion or caught failure and seal files/directories read-only, with
SHA-256 inventory. These permissions prevent accidental writes, not deliberate
modification by the owner. Hard termination/power loss may leave an unsealed run
without `status.json`; such a run is incomplete and must not be treated as results.
Do not alter or delete original data. Keep data, caches and all runs out of Git.
Only code, configs, tests and documentation are versioned. A dirty Git tree is
allowed because its actual source snapshot and diff are preserved per run.

Reproducibility means traceable inputs and procedure; bitwise equality across
hardware/backends is not promised. The manifest covers the entire dataset even
for smoke runs; `limit_per_sequence` and `pairs.csv` record the actual subset.

For the inspected dataset: LF/MF/HF each have 5,376 grayscale 1024×208 JPEGs,
producing 16,125 pairs and approximately 27.5 GB of raw arrays, plus visualizations.
A full run should follow successful real-weight smoke inference and review of
its measured timing. The runner checks disk space before creating a run.

A standalone manifest can be created without loading model weights:

```sh
python scripts/build_manifest.py --output runs/manifest.json
```

That command refuses to overwrite an existing output. Research results must be
interpreted separately from the synthetic integration tests, whose model is a
clearly labeled test double.

## Linux/NVIDIA configuration

`configs/raft_baseline_cuda.yaml` is the same baseline with `device: cuda`.
The default Mac config remains MPS-only. Both refuse CPU fallback. See
[TRANSFER.md](TRANSFER.md) for GPU-specific environment setup and validation.
