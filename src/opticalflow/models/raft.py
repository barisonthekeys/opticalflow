"""Explicit pretrained torchvision RAFT-Large, float32 inference."""

import os
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from opticalflow.data import digest


def choose_device(requested):
    if requested not in ("mps", "cuda"):
        raise ValueError(
            "Experiment 001 requires MPS or CUDA; CPU inference is disabled"
        )
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this process. No CPU fallback or inference was started."
            )
        return "cuda"
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError(
            "Unset PYTORCH_ENABLE_MPS_FALLBACK: CPU operator fallback is disabled"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable in this process. No CPU fallback or inference was started."
        )
    return "mps"


class RaftBaseline:
    def __init__(self, config, cache_root):
        self.device = choose_device(config["device"])
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        torch.hub.set_dir(str(cache_root))
        self.weights = Raft_Large_Weights.C_T_SKHT_V2
        self.model = (
            raft_large(weights=self.weights, progress=True).eval().to(self.device)
        )
        self.transforms = self.weights.transforms()
        self.updates = config["num_flow_updates"]
        checkpoint = (
            Path(cache_root)
            / "checkpoints"
            / Path(urlparse(self.weights.url).path).name
        )
        self.metadata = {
            "device": self.device,
            "device_name": torch.cuda.get_device_name()
            if self.device == "cuda"
            else "Apple MPS",
            "cuda_runtime": torch.version.cuda if self.device == "cuda" else None,
            "weights": self.weights.name,
            "weights_url": self.weights.url,
            "checkpoint_sha256": digest(checkpoint.read_bytes()),
        }

    @torch.inference_mode()
    def predict(self, first, second):
        first, second = self.transforms(first, second)
        flow = self.model(
            first.to(self.device), second.to(self.device), num_flow_updates=self.updates
        )[-1]
        return flow[0].cpu()
