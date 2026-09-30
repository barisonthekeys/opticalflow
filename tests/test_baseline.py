import json
import stat
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from opticalflow import experiment
from opticalflow.data import build_manifest, digest, load_frame
from opticalflow.inference import infer, statistics
from opticalflow.preprocessing import prepare


def frames(tmp_path, numbers=(9, 10, 11)):
    directory = tmp_path / "data" / "LF"
    directory.mkdir(parents=True)
    for number in numbers:
        Image.fromarray(np.full((129, 131), number, dtype=np.uint8)).save(
            directory / f"frame_{number}.png"
        )
    return tmp_path / "data"


def test_order_and_content_integrity(tmp_path):
    root = frames(tmp_path)
    manifest = build_manifest(root, ["LF"])
    rows = manifest["sequences"]["LF"]
    assert [r["frame"] for r in rows] == [9, 10, 11]
    assert load_frame(root, rows[0]).shape == (129, 131, 3)
    (root / rows[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_frame(root, rows[0])


def test_gaps_rejected(tmp_path):
    root = frames(tmp_path, (0, 2))
    with pytest.raises(ValueError, match="missing"):
        build_manifest(root, ["LF"])


def test_padding_and_uint16():
    array = np.full((129, 131, 3), 255, np.uint8)
    tensor = prepare(array)
    assert tensor.shape == (1, 3, 136, 136)
    assert torch.all(tensor == 1)
    assert prepare(np.zeros((17, 19, 3), np.uint8)).shape == (1, 3, 128, 128)
    with pytest.raises(ValueError, match="uint16"):
        prepare(array.astype(np.uint16))
    assert torch.all(prepare(np.full((128, 128, 3), 4095, np.uint16), 4095) == 1)


class FakeModel:
    def __init__(self, *args):
        self.metadata = {"test_double": True}

    def predict(self, first, second):
        flow = torch.zeros((2, *first.shape[-2:]))
        flow[0] = 3
        flow[1] = 4
        return flow


def test_flow_crop_units_and_statistics():
    array = np.zeros((129, 131, 3), np.uint8)
    flow = infer(FakeModel(), array, array, None)
    assert flow.shape == (129, 131, 2)
    assert flow.dtype == np.float32
    stats = statistics(flow)
    assert stats["mean_u"] == 3 and stats["mean_v"] == 4
    assert stats["mean_magnitude"] == 5 and stats["std_magnitude"] == 0


def unseal(root):
    for path in [root, *root.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def test_run_provenance_and_no_overwrite(tmp_path, monkeypatch):
    frames(tmp_path)
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/raft_baseline.yaml").read_text()
    )
    config["sequences"] = ["LF"]
    monkeypatch.setattr(experiment, "RaftBaseline", FakeModel)
    # Keep real source/git/environment capture in this integration check.
    first = experiment.run(config, tmp_path, limit=1)
    second = experiment.run(config, tmp_path, limit=1)
    try:
        assert first != second
        assert json.loads((first / "status.json").read_text())["completed_pairs"] == 1
        flow = np.load(first / "LF/000009_000010.npy")
        assert flow.shape == (129, 131, 2) and flow.dtype == np.float32
        assert not (first.stat().st_mode & stat.S_IWUSR)
        for filename, expected in json.loads(
            (first / "checksums.json").read_text()
        ).items():
            assert digest((first / filename).read_bytes()) == expected
        assert (
            json.loads((first / "summary.json").read_text())["LF"]["mean_magnitude"]
            == 5
        )
    finally:
        unseal(first)
        unseal(second)


def test_failed_run_is_labeled(tmp_path, monkeypatch):
    frames(tmp_path)
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/raft_baseline.yaml").read_text()
    )
    config["sequences"] = ["LF"]

    def fail(*args):
        raise RuntimeError("test model failure")

    monkeypatch.setattr(experiment, "RaftBaseline", fail)
    with pytest.raises(RuntimeError, match="test model"):
        experiment.run(config, tmp_path, limit=1)
    output = next((tmp_path / "runs").iterdir())
    try:
        status = json.loads((output / "status.json").read_text())
        assert status["status"] == "failed" and status["completed_pairs"] == 0
    finally:
        unseal(output)


def test_mps_required_no_cpu_fallback(monkeypatch):
    from opticalflow.models.raft import choose_device

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        choose_device("mps")
    with pytest.raises(ValueError, match="CPU inference is disabled"):
        choose_device("cpu")
    with pytest.raises(ValueError, match="CPU inference is disabled"):
        choose_device("auto")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert choose_device("mps") == "mps"
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(RuntimeError, match="operator fallback is disabled"):
        choose_device("mps")


def test_cuda_requires_available_gpu(monkeypatch):
    from opticalflow.models.raft import choose_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        choose_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert choose_device("cuda") == "cuda"


def test_cuda_config_only_changes_device():
    directory = Path(__file__).parents[1] / "configs"
    mac = yaml.safe_load((directory / "raft_baseline.yaml").read_text())
    linux = yaml.safe_load((directory / "raft_baseline_cuda.yaml").read_text())
    experiment.validate(linux)
    assert linux.pop("device") == "cuda"
    assert mac.pop("device") == "mps"
    assert linux == mac
