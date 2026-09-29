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
