"""Tests for scripts/09_visualize_predictions.py's non-rendering logic.

Two review findings are pinned here, both about figures that would otherwise be
produced without complaint from inputs the model never saw:

1. ``CHDDataset`` must be built with ``require_pose=not config.no_pose``, the
   same rule ``chd/eval/predict.py`` and ``train.py`` use. A ``--no-pose`` run
   on a dataset that never precomputed a pose cache otherwise survives
   ``predict_run`` and only dies inside ``gather_records``.
2. Every ``--also-run`` model is fed the *primary* run's image tensors, so a
   run trained at a different ``img_size`` or on a different dataset would get
   an out-of-distribution forward pass rendered as an ablation column.

The script is not an importable package, so it is loaded by path.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chd.models.factory import build_model  # noqa: E402
from test_eval_predict import IMG_SIZE, TINY_ARGS, make_dataset  # noqa: E402


def load_script() -> types.ModuleType:
    path = ROOT / "scripts" / "09_visualize_predictions.py"
    spec = importlib.util.spec_from_file_location("script_09", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> types.ModuleType:
    return load_script()


def write_checkpoint(path: Path, **overrides) -> None:
    args = {**TINY_ARGS, "data_root": "data", **overrides}
    cfg = argparse.Namespace(**args, no_pretrained=True)
    model = build_model(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 1, "best_s_alpha": 0.5, "model": model.state_dict(),
                "ema": None, "args": args}, path)


def fake_bundle(name: str, dataset: str, img_size: int):
    return types.SimpleNamespace(
        name=name, config=argparse.Namespace(dataset=dataset, img_size=img_size))


class TestRequirePose:
    def test_no_pose_run_does_not_require_a_pose_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script,
    ) -> None:
        """require_pose must follow config.no_pose, exactly as predict.py does.

        CHDDataset is intercepted so the assertion is about the constructor
        call itself rather than about rendering a figure, which keeps the test
        cheap. A missing pose cache on disk would make the real constructor
        raise FileNotFoundError, which is the bug.
        """
        data_root = tmp_path / "data"
        make_dataset(data_root / "toy", write_pose=False)
        write_checkpoint(tmp_path / "runs" / "toy" / "best.pth",
                         dataset="toy", no_pose=True, data_root=str(data_root))

        seen: dict = {}
        real = script.CHDDataset

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(script, "CHDDataset", spy)
        monkeypatch.setattr(sys, "argv", [
            "09", "--run", "toy", "--runs-root", str(tmp_path / "runs"),
            "--data-root", str(data_root), "--num-images", "1",
            "--out", str(tmp_path / "figs"),
        ])
        script.main()
        assert seen["require_pose"] is False

    def test_pose_run_still_requires_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script,
    ) -> None:
        data_root = tmp_path / "data"
        make_dataset(data_root / "toy")
        write_checkpoint(tmp_path / "runs" / "toy" / "best.pth",
                         dataset="toy", no_pose=False, data_root=str(data_root))

        seen: dict = {}
        real = script.CHDDataset

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(script, "CHDDataset", spy)
        monkeypatch.setattr(sys, "argv", [
            "09", "--run", "toy", "--runs-root", str(tmp_path / "runs"),
            "--data-root", str(data_root), "--num-images", "1",
            "--out", str(tmp_path / "figs"),
        ])
        script.main()
        assert seen["require_pose"] is True


class TestAblationCompatibility:
    def test_matching_runs_are_accepted(self, script) -> None:
        primary = fake_bundle("a", "acd1k", IMG_SIZE)
        script.check_ablation_compatible(primary, [fake_bundle("b", "acd1k", IMG_SIZE)])

    def test_no_extra_runs_is_always_fine(self, script) -> None:
        script.check_ablation_compatible(fake_bundle("a", "acd1k", IMG_SIZE), [])

    def test_different_img_size_is_refused(self, script) -> None:
        with pytest.raises(SystemExit) as excinfo:
            script.check_ablation_compatible(
                fake_bundle("a", "acd1k", 640), [fake_bundle("b", "acd1k", 352)])
        assert "img_size" in str(excinfo.value) and "b" in str(excinfo.value)

    def test_different_dataset_is_refused(self, script) -> None:
        with pytest.raises(SystemExit) as excinfo:
            script.check_ablation_compatible(
                fake_bundle("a", "acd1k", 640), [fake_bundle("b", "cpd1k", 640)])
        assert "cpd1k" in str(excinfo.value)


class TestResizeReuse:
    def test_gather_records_uses_the_metric_paths_resize(self, script) -> None:
        """09 must not fork its own cv2.resize of the probability map.

        The rendered map and the scored map have to come from one function or
        the two can drift (argument order, clipping) without anything failing.
        """
        from chd.eval import predict

        assert script.resize_prob is predict.resize_prob
