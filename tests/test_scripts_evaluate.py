"""Tests for scripts/08_evaluate.py's crash-safety behaviour.

A full ``combined`` evaluation is ~40 CPU-minutes of metric work. A worker
dying at image 1100 of 1150 used to discard every completed row, so the pinned
behaviour is: flush what finished to ``per_image.csv``, say so on stderr, and
still re-raise — and deliberately do *not* write ``summary.json`` or
``metrics.md``, so a partial run can never be mistaken for a finished one.

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
from test_eval_predict import TINY_ARGS, make_dataset  # noqa: E402

STEMS = tuple(f"s{i}" for i in range(6))


def load_script() -> types.ModuleType:
    path = ROOT / "scripts" / "08_evaluate.py"
    spec = importlib.util.spec_from_file_location("script_08", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> types.ModuleType:
    return load_script()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    make_dataset(tmp_path / "data" / "toy", stems=STEMS)
    args = {**TINY_ARGS, "data_root": str(tmp_path / "data")}
    cfg = argparse.Namespace(**args, no_pretrained=True)
    ckpt = tmp_path / "runs" / "toy" / "best.pth"
    ckpt.parent.mkdir(parents=True)
    torch.save({"epoch": 1, "best_s_alpha": 0.1, "model": build_model(cfg).state_dict(),
                "ema": None, "args": args}, ckpt)
    return tmp_path


def run_args(workspace: Path) -> list[str]:
    return ["08", "--run", "toy", "--runs-root", str(workspace / "runs"),
            "--data-root", str(workspace / "data"), "--workers", "0",
            "--device", "cpu", "--out", str(workspace / "out")]


class TestPartialFlush:
    def test_completed_rows_survive_a_mid_run_failure(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys, script,
    ) -> None:
        real = script.metric_row
        seen = {"n": 0}

        def boom(pred):
            seen["n"] += 1
            if seen["n"] > 3:
                raise RuntimeError("simulated worker crash")
            return real(pred)

        monkeypatch.setattr(script, "metric_row", boom)
        monkeypatch.setattr(sys, "argv", run_args(workspace))
        with pytest.raises(RuntimeError):
            script.main()

        out = workspace / "out"
        body = (out / "per_image.csv").read_text().strip().splitlines()
        assert len(body) - 1 == 3  # header + the three rows that completed
        assert "PARTIAL results flushed" in capsys.readouterr().err

    def test_a_partial_run_writes_no_summary_or_metrics(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch, script,
    ) -> None:
        def boom(pred):
            raise RuntimeError("simulated worker crash")

        monkeypatch.setattr(script, "metric_row", boom)
        monkeypatch.setattr(sys, "argv", run_args(workspace))
        with pytest.raises(RuntimeError):
            script.main()

        out = workspace / "out"
        assert not (out / "summary.json").exists()
        assert not (out / "metrics.md").exists()

    def test_a_clean_run_writes_the_full_report_set(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys, script,
    ) -> None:
        monkeypatch.setattr(sys, "argv", run_args(workspace))
        script.main()
        out = workspace / "out"
        for name in ("per_image.csv", "failures.csv", "summary.json", "metrics.md"):
            assert (out / name).exists(), name
        # The toy dataset is all-positive, so the gate warning must fire.
        assert "single-class split" in capsys.readouterr().out
