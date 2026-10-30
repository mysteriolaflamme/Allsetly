"""20-pair evaluation: run investigations and write RESULTS.md (uses the network).

Run from the project root: ``python -m eval.run_eval``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from temoin.agent import run_investigation
from temoin.config import ConfigError, Settings, load_settings
from temoin.prompts import AGENT_PROMPT_VERSION, COMPARE_PROMPT_VERSION, PROMPT_VERSION
from temoin.schemas import Comparison, TrailEntry

VERDICTS = ("CONFORME", "ENDOMMAGÉ", "À VÉRIFIER", "NON CONFORME")
VERDICTS_BY_LENGTH = ("NON CONFORME", "ENDOMMAGÉ", "À VÉRIFIER", "CONFORME")


class EvalError(Exception):
    """Raised when the evaluation set or settings are unusable."""


@dataclass(frozen=True)
class EvalPair:
    """One annotated sale/delivery pair with its expected verdict."""

    pair_id: str
    sale: Path
    delivery: Path
    sale_code: str | None
    delivery_code: str | None
    relation: str
    codes_ok: bool
    expected: str
    notes: str


@dataclass(frozen=True)
class PairRecord:
    """Outcome of one evaluated pair (or its error)."""

    pair_id: str
    relation: str
    expected: str
    verdict: str
    model_verdict: str | None
    confidence: int
    seconds: float
    cached: bool = False
    error: str | None = None


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate scores over finished pairs."""

    total: int
    errors: int
    correct: int
    accuracy: float
    confusion: dict[str, dict[str, int]]
    false_accusations: tuple[str, ...]
    false_accusation_rate: float
    fraud_confirmed: tuple[str, ...]
    fraud_cautious: tuple[str, ...]
    fraud_missed: tuple[str, ...]
    model_hits_on_different: int


_REQUIRED_COLUMNS = (
    "pair_id",
    "sale",
    "delivery",
    "sale_code",
    "delivery_code",
    "relation",
    "codes",
    "expected",
    "notes",
)


def parse_labels(labels_path: Path, pairs_dir: Path) -> list[EvalPair]:
    """Load and validate the annotated pair list."""
    if not labels_path.is_file():
        raise EvalError(f"Labels introuvables : « {labels_path} ».")
    try:
        text = labels_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EvalError(f"Labels illisibles : « {labels_path} ».") from error
    reader = csv.DictReader(text.splitlines())
    missing = [name for name in _REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
    if missing:
        raise EvalError(f"Colonnes manquantes dans labels.csv : {', '.join(missing)}.")
    pairs: list[EvalPair] = []
    seen: set[str] = set()
    for line_no, row in enumerate(reader, start=2):
        pair_id = (row["pair_id"] or "").strip()
        if not pair_id:
            raise EvalError(f"Ligne {line_no} : pair_id manquant.")
        if pair_id in seen:
            raise EvalError(f"Ligne {line_no} : pair_id dupliqué « {pair_id} ».")
        seen.add(pair_id)
        relation = (row["relation"] or "").strip()
        if relation not in ("same", "different"):
            raise EvalError(f"Ligne {line_no} : relation « {relation} » invalide.")
        codes = (row["codes"] or "").strip()
        if codes not in ("ok", "missing"):
            raise EvalError(f"Ligne {line_no} : codes « {codes} » invalide.")
        expected = (row["expected"] or "").strip()
        if expected not in VERDICTS:
            raise EvalError(f"Ligne {line_no} : verdict « {expected} » invalide.")
        sale = pairs_dir / (row["sale"] or "").strip()
        delivery = pairs_dir / (row["delivery"] or "").strip()
        if not sale.is_file():
            raise EvalError(f"Ligne {line_no} : photo introuvable « {sale} ».")
        if not delivery.is_file():
            raise EvalError(f"Ligne {line_no} : photo introuvable « {delivery} ».")
        pairs.append(
            EvalPair(
                pair_id=pair_id,
                sale=sale,
                delivery=delivery,
                sale_code=(row["sale_code"] or "").strip() or None,
                delivery_code=(row["delivery_code"] or "").strip() or None,
                relation=relation,
                codes_ok=codes == "ok",
                expected=expected,
                notes=(row["notes"] or "").strip(),
            )
        )
    if not pairs:
        raise EvalError("Aucune paire dans labels.csv.")
    return pairs


def model_verdict_from_trail(trail: list[TrailEntry]) -> str | None:
    """Return the model's raw verdict from the trail, if any."""
    for entry in reversed(trail):
        if entry.action != "verdict":
            continue
        for verdict in VERDICTS_BY_LENGTH:
            if verdict in entry.summary:
                return verdict
    return None


def _record_from(
    pair: EvalPair, comparison: Comparison, *, seconds: float, cached: bool
) -> PairRecord:
    """Build a pair record from a finished comparison."""
    return PairRecord(
        pair_id=pair.pair_id,
        relation=pair.relation,
        expected=pair.expected,
        verdict=comparison.verdict,
        model_verdict=model_verdict_from_trail(comparison.trail),
        confidence=comparison.confidence,
        seconds=seconds,
        cached=cached,
        error=None,
    )


def run_pair(
    pair: EvalPair,
    settings: Settings,
    results_dir: Path,
    *,
    investigate: Callable[..., Comparison] = run_investigation,
    refresh: bool = False,
) -> PairRecord:
    """Run one pair (resuming cached results) and return its record."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cached_path = results_dir / f"{pair.pair_id}.json"
    if cached_path.is_file() and not refresh:
        try:
            comparison = Comparison.model_validate_json(cached_path.read_text(encoding="utf-8"))
        except ValueError:
            comparison = None
        if comparison is not None:
            return _record_from(pair, comparison, seconds=0.0, cached=True)
    started = time.monotonic()
    try:
        comparison = investigate(
            pair.sale,
            pair.delivery,
            sale_code=pair.sale_code,
            delivery_code=pair.delivery_code,
            settings=settings,
        )
    except Exception as error:
        seconds = time.monotonic() - started
        return PairRecord(
            pair_id=pair.pair_id,
            relation=pair.relation,
            expected=pair.expected,
            verdict="",
            model_verdict=None,
            confidence=0,
            seconds=seconds,
            cached=False,
            error=str(error)[:300],
        )
    seconds = time.monotonic() - started
    cached_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    return _record_from(pair, comparison, seconds=seconds, cached=False)


def compute_metrics(records: list[PairRecord]) -> EvalMetrics:
    """Compute accuracy, confusion and fraud rates over finished pairs."""
    done = [record for record in records if record.error is None]
    correct = sum(1 for record in done if record.verdict == record.expected)
    confusion = {expected: {got: 0 for got in VERDICTS} for expected in VERDICTS}
    for record in done:
        if record.verdict in VERDICTS:
            confusion[record.expected][record.verdict] += 1
    same = [record for record in done if record.relation == "same"]
    false_accusations = tuple(record.pair_id for record in same if record.verdict == "NON CONFORME")
    different = [record for record in done if record.relation == "different"]
    confirmed = tuple(record.pair_id for record in different if record.verdict == "NON CONFORME")
    cautious = tuple(record.pair_id for record in different if record.verdict == "À VÉRIFIER")
    missed = tuple(
        record.pair_id for record in different if record.verdict in ("CONFORME", "ENDOMMAGÉ")
    )
    model_hits = sum(1 for record in different if record.model_verdict == "NON CONFORME")
    return EvalMetrics(
        total=len(records),
        errors=len(records) - len(done),
        correct=correct,
        accuracy=(correct / len(done)) if done else 0.0,
        confusion=confusion,
        false_accusations=false_accusations,
        false_accusation_rate=(len(false_accusations) / len(same)) if same else 0.0,
        fraud_confirmed=confirmed,
        fraud_cautious=cautious,
        fraud_missed=missed,
        model_hits_on_different=model_hits,
    )


def render_results(
    records: list[PairRecord], metrics: EvalMetrics, *, model: str, run_at: str, command: str
) -> str:
    """Render RESULTS.md from records and metrics."""
    scored = metrics.total - metrics.errors
    lines = [
        "# Témoin — Evaluation results",
        "",
        f"Run at {run_at} with `{model}` ",
        f"(constat v{PROMPT_VERSION}, compare v{COMPARE_PROMPT_VERSION}, "
        f"agent v{AGENT_PROMPT_VERSION}).",
        "",
        f"Command: `{command}`",
        "",
        "## Headline",
        "",
        f"- Pairs: {metrics.total} ({metrics.errors} errors)",
        f"- Accuracy: **{metrics.correct}/{scored} ({metrics.accuracy:.0%})**",
        "- False accusations: "
        f"**{len(metrics.false_accusations)} "
        f"(rate {metrics.false_accusation_rate:.0%} over same-object pairs)** — goal: 0 %",
        f"- Fraud confirmed: {len(metrics.fraud_confirmed)}, "
        f"cautious: {len(metrics.fraud_cautious)}, missed: {len(metrics.fraud_missed)}",
        f"- Model raw verdicts naming a different object: {metrics.model_hits_on_different}",
        "",
        "## Confusion matrix (expected × predicted)",
        "",
        "| expected \\ predicted | " + " | ".join(VERDICTS) + " |",
        "|" + "---|" * (len(VERDICTS) + 1),
    ]
    for expected in VERDICTS:
        row = metrics.confusion[expected]
        lines.append(f"| {expected} | " + " | ".join(str(row[got]) for got in VERDICTS) + " |")
    lines += [
        "",
        "## Pairs",
        "",
        "| pair | relation | expected | verdict | model | conf | time |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        verdict_cell = record.verdict if record.error is None else f"ERROR: {record.error}"
        lines.append(
            f"| {record.pair_id} | {record.relation} | {record.expected} | "
            f"{verdict_cell} | {record.model_verdict or '—'} | "
            f"{record.confidence} | {record.seconds:.0f}s |"
        )
    lines += [
        "",
        "## Known gaps",
        "",
        "- Damaged-between-photos (ENDOMMAGÉ): 0 pairs — needs longitudinal photos "
        "of the same object before/after damage.",
        "- Same-model-different-unit: 0 pairs — needs two physical units of one model.",
        "- Add pairs by dropping two photos under `eval/pairs/<id>/` and one row "
        "in `eval/labels.csv`, then rerun.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(prog="eval", description="Run the Témoin pair evaluation.")
    parser.add_argument("--labels", default="eval/labels.csv")
    parser.add_argument("--pairs", default="eval/pairs")
    parser.add_argument("--results", default="eval/results")
    parser.add_argument("--out", default="eval/RESULTS.md")
    parser.add_argument("--limit", type=int, default=None, help="Run the first N pairs only.")
    parser.add_argument("--only", action="append", default=None, help="Run one pair id.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached results.")
    return parser


def main(
    argv: list[str] | None = None,
    investigate: Callable[..., Comparison] = run_investigation,
) -> int:
    """Run the evaluation, write RESULTS.md and return the exit code."""
    args = build_parser().parse_args(argv)
    try:
        pairs = parse_labels(Path(args.labels), Path(args.pairs))
        if args.only:
            wanted = set(args.only)
            pairs = [pair for pair in pairs if pair.pair_id in wanted]
            if not pairs:
                raise EvalError("Aucune paire ne correspond à --only.")
        if args.limit is not None:
            pairs = pairs[: args.limit]
        settings = load_settings()
    except (EvalError, ConfigError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    records: list[PairRecord] = []
    for index, pair in enumerate(pairs, start=1):
        record = run_pair(
            pair, settings, Path(args.results), investigate=investigate, refresh=args.refresh
        )
        records.append(record)
        if record.error is not None:
            status = f"ERROR {record.error}"
        else:
            status = f"{record.verdict} (model {record.model_verdict}, conf {record.confidence})"
        print(
            f"[{index}/{len(pairs)}] {pair.pair_id}: {status} [{record.seconds:.0f}s]", flush=True
        )
    metrics = compute_metrics(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shown = argv if argv is not None else sys.argv[1:]
    command = " ".join(["python -m eval.run_eval", *shown])
    out.write_text(
        render_results(
            records,
            metrics,
            model=settings.model_name,
            run_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            command=command,
        ),
        encoding="utf-8",
    )
    print(f"Accuracy: {metrics.correct}/{metrics.total - metrics.errors}", flush=True)
    print(f"False accusations: {len(metrics.false_accusations)}", flush=True)
    print(f"Results written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
