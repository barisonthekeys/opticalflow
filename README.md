# Optical Flow for Flow Boiling

Optical-flow analysis of high-speed flow-boiling imagery.

## Data

Three experimental sequences:

- LF: low heat flux
- MF: medium heat flux
- HF: high heat flux

Raw image data are not tracked by Git.

## Environment

Conda environment: `of`

Environment specifications are stored in:

- `environment/environment.yml`
- `environment/requirements-lock.txt`
- `environment/system_info.txt`

## Project structure

- `configs/` experiment configurations
- `data/` raw image sequences
- `src/opticalflow/` reusable project code
- `scripts/` experiment entry points
- `runs/` generated experiment outputs
- `notebooks/` exploratory analysis
- `tests/` automated checks

## Experiment 001

The reusable pretrained RAFT-Large baseline is documented in [EXPERIMENT001.md](EXPERIMENT001.md). Run `python scripts/run_experiment.py --limit 2` in Conda environment `of` for a six-pair smoke test, then omit `--limit` for all consecutive LF/MF/HF pairs. Each run records its exact inputs, configuration, code, environment, and outputs.

## Another computer

See [TRANSFER.md](TRANSFER.md) for Linux/NVIDIA setup, CUDA smoke testing, and
transferring inputs separately. The default config remains Apple MPS; Linux uses
`configs/raft_baseline_cuda.yaml`. Both prohibit CPU inference fallback.
