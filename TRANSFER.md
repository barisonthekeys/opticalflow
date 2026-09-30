# Clone and run on another computer

This repository contains the experiment code, configuration, tests and original
Mac environment records. Raw images, flow arrays, runs, videos and downloaded
weights are excluded. Transfer LF/MF/HF inputs separately into `data/LF`,
`data/MF`, `data/HF`, preserving filenames.

## Linux with an NVIDIA GPU

Clone this repository using its GitHub Clone URL, then enter the checkout.
Create the portable environment (the original `environment.yml` is a Mac snapshot):

```sh
conda env create -f environment/environment-linux.yml
conda activate of
nvidia-smi
```

Install a CUDA-enabled PyTorch build compatible with the GPU and installed driver.
The original experiment used **torch 2.14.0 and torchvision 0.29.0**. Select an
appropriate CUDA wheel channel using [PyTorch's official installer](https://pytorch.org/get-started/locally/)
and retain these versions where supported. Replace CUDA_WHEEL_URL below with the
official CUDA index appropriate to the machine:

```sh
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url CUDA_WHEEL_URL
python -m pip check
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name()); print(torch.ones(1, device='cuda'))"
python -m pytest -q
```

Resolve driver/GPU incompatibilities before inference. Do not choose CPU wheels
or enable CPU fallback. If those versions cannot support this GPU, record any
version changes as a distinct environment. The Linux bootstrap intentionally
leaves PyTorch to this hardware-specific step; it is not an exact Linux lock.

After transferring the raw images:

```sh
python scripts/run_experiment.py --config configs/raft_baseline_cuda.yaml --limit 2
# Review the six-pair smoke run before starting the full experiment:
python scripts/run_experiment.py --config configs/raft_baseline_cuda.yaml
```

The CUDA config changes only the device. Model, C_T_SKHT_V2 weights, stride,
normalization, padding, update count, outputs and visualization scale match the
Mac baseline. `CUDA_VISIBLE_DEVICES` can select which NVIDIA GPU the process sees.
CUDA inference cannot be tested on this Mac; selection/availability guards are
unit-tested, and real GPU validation must happen on the Linux computer. Each run
captures its software versions, CUDA runtime and device. Results need not be
bitwise identical across GPU backends.

## Apple-silicon Mac

The existing `configs/raft_baseline.yaml` continues to require MPS. Environment
files without a Linux suffix preserve the original environment record, including
its prefix and machine information. Adjust the prefix when recreating it at a
different location.

```sh
conda activate of
python scripts/run_experiment.py --limit 2
```

Both configurations fail rather than falling back to CPU. A clone creates a new
local checkout; later changes can be received with `git pull` after preserving
local work. Existing experiment runs are never overwritten or modified.
