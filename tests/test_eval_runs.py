"""Tests for chd.eval.runs — recovering a run's configuration from its checkpoint.

The real checkpoints live on a different machine, so every test here builds a
synthetic checkpoint with the same structure train.py writes (see train.py's
checkpoint dict, which stores "args": vars(args)). That structure is the
contract these tests pin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval import runs  # noqa: E402
from chd.models.factory import build_model  # noqa: E402

TINY_ARGS = {
    "architecture": "chdnet",
    "backbone": "tiny_test",
    "dataset": "acd1k",
    "img_size": 64,
    "os_streams": 2,
    "no_pose": False,
    "data_root": Path("data"),
}


def write_checkpoint(path: Path, *, with_ema: bool, args: dict | None = None) -> None:
    """Build a real state_dict from the tiny_test backbone so load_state_dict is exercised."""
    cfg = argparse.Namespace(**{**TINY_ARGS, **(args or {})}, no_pretrained=True)
    model = build_model(cfg)
    state = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": 7,
        "best_s_alpha": 0.5,
        "model": state,
        "ema": {k: v.clone() for k, v in state.items()} if with_ema else None,
        "args": {**TINY_ARGS, **(args or {})},
    }, path)


class TestResolveCheckpoint:
    def test_prefers_best_over_last(self, tmp_path: Path) -> None:
        (tmp_path / "r1").mkdir()
        (tmp_path / "r1" / "best.pth").touch()
        (tmp_path / "r1" / "last.pth").touch()
        assert runs.resolve_checkpoint("r1", tmp_path).name == "best.pth"

    def test_falls_back_to_last_when_best_missing(self, tmp_path: Path) -> None:
        (tmp_path / "r1").mkdir()
        (tmp_path / "r1" / "last.pth").touch()
        assert runs.resolve_checkpoint("r1", tmp_path).name == "last.pth"

    def test_missing_run_lists_available_names(self, tmp_path: Path) -> None:
        (tmp_path / "camo-human-final").mkdir()
        (tmp_path / "camo-human-final" / "best.pth").touch()
        with pytest.raises(FileNotFoundError, match="camo-human-final"):
            runs.resolve_checkpoint("nope", tmp_path)

    def test_run_dir_without_any_checkpoint_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            runs.resolve_checkpoint("empty", tmp_path)


class TestAvailableRuns:
    def test_lists_only_dirs_holding_a_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "good").mkdir()
        (tmp_path / "good" / "best.pth").touch()
        (tmp_path / "bare").mkdir()
        assert runs.available_runs(tmp_path) == ["good"]

    def test_missing_root_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert runs.available_runs(tmp_path / "absent") == []


class TestConfigFromCheckpoint:
    def test_recovers_stored_args(self) -> None:
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)})
        assert cfg.dataset == "acd1k"
        assert cfg.img_size == 64
        assert cfg.backbone == "tiny_test"

    def test_overrides_win(self) -> None:
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)}, overrides={"img_size": 128})
        assert cfg.img_size == 128

    def test_none_valued_overrides_are_ignored(self) -> None:
        """Unset CLI flags arrive as None and must not clobber the stored config."""
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)}, overrides={"img_size": None})
        assert cfg.img_size == 64

    def test_missing_args_is_an_actionable_error(self) -> None:
        with pytest.raises(KeyError, match="args"):
            runs.config_from_checkpoint({"model": {}})

    def test_pretraining_is_disabled_so_no_weights_are_downloaded(self) -> None:
        """Checkpoint weights overwrite the backbone anyway; downloading ImageNet
        first would only cost time and require network access."""
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)})
        assert cfg.no_pretrained is True


class TestLoadRun:
    def test_prefers_ema_weights(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path / "r1" / "best.pth", with_ema=True)
        bundle = runs.load_run("r1", runs_root=tmp_path)
        assert bundle.weights == "ema"

    def test_falls_back_to_raw_weights(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path / "r1" / "best.pth", with_ema=False)
        bundle = runs.load_run("r1", runs_root=tmp_path)
        assert bundle.weights == "raw"

    def test_model_is_in_eval_mode_and_carries_metadata(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path / "r1" / "best.pth", with_ema=True)
        bundle = runs.load_run("r1", runs_root=tmp_path)
        assert bundle.model.training is False
        assert bundle.epoch == 7
        assert bundle.best_s_alpha == pytest.approx(0.5)
        assert bundle.name == "r1"

    def test_folder_name_does_not_determine_dataset(self, tmp_path: Path) -> None:
        """The real runs include 'camo-human-final' for dataset 'camo_human'."""
        write_checkpoint(tmp_path / "camo-human-final" / "best.pth",
                         with_ema=True, args={"dataset": "camo_human"})
        bundle = runs.load_run("camo-human-final", runs_root=tmp_path)
        assert bundle.config.dataset == "camo_human"

    def test_explicit_checkpoint_path_bypasses_run_lookup(self, tmp_path: Path) -> None:
        path = tmp_path / "somewhere" / "weights.pth"
        write_checkpoint(path, with_ema=True)
        bundle = runs.load_run("ignored", runs_root=tmp_path / "absent", checkpoint=path)
        assert bundle.checkpoint_path == path
