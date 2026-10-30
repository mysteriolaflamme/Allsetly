"""Tests for the evaluation runner (offline, no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from eval.run_eval import (
    EvalError,
    PairRecord,
    compute_metrics,
    main,
    model_verdict_from_trail,
    parse_labels,
    render_results,
)
from temoin.config import Settings
from temoin.schemas import Comparison, TrailEntry

_HEADER = "pair_id,sale,delivery,sale_code,delivery_code,relation,codes,expected,notes\n"


def _write_labels(path: Path, body: str) -> Path:
    """Write a labels.csv file with the standard header."""
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def _record(
    pair_id: str,
    relation: str,
    expected: str,
    verdict: str,
    model_verdict: str | None = None,
    error: str | None = None,
) -> PairRecord:
    """Build a pair record for metrics tests."""
    return PairRecord(
        pair_id=pair_id,
        relation=relation,
        expected=expected,
        verdict=verdict,
        model_verdict=model_verdict,
        confidence=80,
        seconds=3.0,
        cached=False,
        error=error,
    )


def test_parse_labels_loads_pairs(tmp_path: Path) -> None:
    """Two valid rows load with codes, relation and expectation."""
    pairs_dir = tmp_path / "pairs"
    (pairs_dir / "p1").mkdir(parents=True)
    (pairs_dir / "p1" / "sale.jpg").write_bytes(b"s")
    (pairs_dir / "p1" / "delivery.jpg").write_bytes(b"d")
    labels = _write_labels(
        tmp_path / "labels.csv", "p1,p1/sale.jpg,p1/delivery.jpg,K7,,same,missing,À VÉRIFIER,note\n"
    )
    (pairs := parse_labels(labels, pairs_dir))
    assert len(pairs) == 1
    pair = pairs[0]
    assert (pair.pair_id, pair.sale_code, pair.delivery_code) == ("p1", "K7", None)
    assert (pair.relation, pair.codes_ok, pair.expected) == ("same", False, "À VÉRIFIER")
    assert pair.notes == "note"


def test_parse_labels_rejects_bad_rows(tmp_path: Path) -> None:
    """Bad enums, duplicates and missing photos are plain-language errors."""
    pairs_dir = tmp_path / "pairs"
    (pairs_dir / "p1").mkdir(parents=True)
    (pairs_dir / "p1" / "sale.jpg").write_bytes(b"s")
    (pairs_dir / "p1" / "delivery.jpg").write_bytes(b"d")
    good = "p1,p1/sale.jpg,p1/delivery.jpg,K7,Q9,same,ok,CONFORME,\n"
    cases = [
        ("p1,p1/sale.jpg,p1/delivery.jpg,K7,Q9,close,ok,CONFORME,\n", "relation"),
        ("p1,p1/sale.jpg,p1/delivery.jpg,K7,Q9,same,maybe,CONFORME,\n", "codes"),
        ("p1,p1/sale.jpg,p1/delivery.jpg,K7,Q9,same,ok,MAYBE,\n", "verdict"),
        ("p1,p1/sale.jpg,p1/nope.jpg,K7,Q9,same,ok,CONFORME,\n", "introuvable"),
        (good + good, "dupliqué"),
        ("", "Aucune paire"),
    ]
    for body, fragment in cases:
        labels = _write_labels(tmp_path / "labels.csv", body)
        with pytest.raises(EvalError, match=fragment):
            parse_labels(labels, pairs_dir)
    with pytest.raises(EvalError, match="introuvables"):
        parse_labels(tmp_path / "nope.csv", pairs_dir)
    bad_columns = tmp_path / "columns.csv"
    bad_columns.write_text("pair_id,sale\np1,x\n", encoding="utf-8")
    with pytest.raises(EvalError, match="Colonnes manquantes"):
        parse_labels(bad_columns, pairs_dir)


def test_model_verdict_from_trail() -> None:
    """The raw verdict is read from the last verdict entry, longest match first."""
    assert (
        model_verdict_from_trail(
            [TrailEntry(step=1, action="verdict", summary="Model verdict: NON CONFORME (c 80).")]
        )
        == "NON CONFORME"
    )
    assert (
        model_verdict_from_trail(
            [TrailEntry(step=2, action="verdict", summary="Model verdict: CONFORME (c 90).")]
        )
        == "CONFORME"
    )
    assert (
        model_verdict_from_trail(
            [TrailEntry(step=3, action="verdict", summary="Model verdict: À VÉRIFIER (c 40).")]
        )
        == "À VÉRIFIER"
    )
    assert model_verdict_from_trail([TrailEntry(step=1, action="tool:zoom", summary="x")]) is None


def test_compute_metrics_counts_and_rates() -> None:
    """Accuracy, confusion, false accusations and fraud splits are exact."""
    records = [
        _record("p1", "same", "CONFORME", "CONFORME", "CONFORME"),
        _record("p2", "same", "À VÉRIFIER", "À VÉRIFIER", "CONFORME"),
        _record("p3", "same", "CONFORME", "NON CONFORME", "NON CONFORME"),
        _record("p4", "different", "NON CONFORME", "NON CONFORME", "NON CONFORME"),
        _record("p5", "different", "À VÉRIFIER", "À VÉRIFIER", "NON CONFORME"),
        _record("p6", "different", "À VÉRIFIER", "CONFORME", "CONFORME"),
        _record("p7", "same", "CONFORME", "", None, error="boom"),
    ]
    metrics = compute_metrics(records)
    assert (metrics.total, metrics.errors, metrics.correct) == (7, 1, 4)
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.confusion["CONFORME"]["NON CONFORME"] == 1
    assert metrics.confusion["À VÉRIFIER"]["À VÉRIFIER"] == 2
    assert metrics.false_accusations == ("p3",)
    assert metrics.false_accusation_rate == pytest.approx(1 / 3)
    assert metrics.fraud_confirmed == ("p4",)
    assert metrics.fraud_cautious == ("p5",)
    assert metrics.fraud_missed == ("p6",)
    assert metrics.model_hits_on_different == 2


def test_render_results_contains_sections() -> None:
    """RESULTS.md shows headline numbers, confusion and every pair."""
    records = [
        _record("p1", "same", "CONFORME", "CONFORME", "CONFORME"),
        _record("p2", "same", "CONFORME", "", None, error="boom"),
    ]
    text = render_results(
        records,
        compute_metrics(records),
        model="m",
        run_at="2026-01-01",
        command="run",
    )
    assert "# Témoin — Evaluation results" in text
    assert "Accuracy: **1/1 (100%)**" in text
    assert "False accusations: **0" in text
    assert "| expected \\ predicted |" in text
    assert "| p1 | same | CONFORME | CONFORME | CONFORME | 80 | 3s |" in text
    assert "ERROR: boom" in text
    assert "Known gaps" in text


def test_main_runs_pairs_and_caches_results(
    tmp_path: Path,
    test_settings: Settings,
    comparison: Comparison,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() investigates, writes RESULTS.md and reuses cached pairs."""
    pairs_dir = tmp_path / "pairs"
    (pairs_dir / "p1").mkdir(parents=True)
    (pairs_dir / "p1" / "sale.jpg").write_bytes(b"s")
    (pairs_dir / "p1" / "delivery.jpg").write_bytes(b"d")
    labels = _write_labels(
        tmp_path / "labels.csv", "p1,p1/sale.jpg,p1/delivery.jpg,K7,Q9,same,ok,CONFORME,\n"
    )
    results = tmp_path / "results"
    out = tmp_path / "RESULTS.md"
    calls: list[tuple[str | None, str | None]] = []

    def fake_investigate(
        sale: Path,
        delivery: Path,
        *,
        sale_code: str | None,
        delivery_code: str | None,
        settings: Settings,
    ) -> Comparison:
        calls.append((sale_code, delivery_code))
        return comparison

    monkeypatch.setattr("eval.run_eval.load_settings", lambda *a, **k: test_settings)
    argv = [
        "--labels",
        str(labels),
        "--pairs",
        str(pairs_dir),
        "--results",
        str(results),
        "--out",
        str(out),
    ]
    assert main(argv, investigate=fake_investigate) == 0
    assert calls == [("K7", "Q9")]
    assert "[1/1] p1" in capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert "p1" in text and "Accuracy: **1/1 (100%)**" in text
    assert (results / "p1.json").is_file()
    assert main(argv, investigate=fake_investigate) == 0
    assert len(calls) == 1
    assert main([*argv, "--refresh"], investigate=fake_investigate) == 0
    assert len(calls) == 2
    assert main(["--labels", str(tmp_path / "nope.csv")]) == 1
    monkeypatch.setattr(sys, "argv", ["run_eval", *argv, "--labels", str(tmp_path / "nope.csv")])
    assert main(None, investigate=fake_investigate) == 1
