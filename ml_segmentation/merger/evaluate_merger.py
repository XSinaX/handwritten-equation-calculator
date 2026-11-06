from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ml_segmentation.segmentation.evaluate_segmentation import analyze, equation_flags
from ml_segmentation.merger.merger import groups_to_segments, TOLERANCE


def missed_merge_status(groups: list[list[int]], owners: list[int]) -> tuple[bool, int, int]:
    """Return whether any real symbol remains split across final groups."""
    symbol_owners = sorted({owner for owner in owners if owner > 0})
    missed_instances = 0
    extra_symbol_groups = 0
    for owner in symbol_owners:
        group_count = sum(1 for group in groups if any(owners[i] == owner for i in group))
        if group_count > 1:
            missed_instances += 1
            extra_symbol_groups += group_count - 1
    return missed_instances > 0, missed_instances, extra_symbol_groups


def evaluate_equation(
    *,
    idx: int,
    source: str,
    equation: str,
    image,
    gt,
    noise,
    annotations: list,
    candidate: dict,
    groups: list[list[int]],
    split: str = "test",
) -> dict:
    """Score grouping under the original wrong/missed merge definitions."""
    final_segments = groups_to_segments(candidate["segments"], groups)
    final_metrics = analyze(
        image,
        gt,
        noise,
        annotations,
        final_segments,
        TOLERANCE["pixel_tolerance"],
        TOLERANCE["component_fraction"],
        TOLERANCE["symbol_fraction"],
    )
    missed, missed_instances, extra_symbol_groups = missed_merge_status(groups, candidate["owners"])
    wrong_groups = int(final_metrics["wrong_merge_components_significant"])
    wrong = wrong_groups > 0
    recoverable = bool(candidate["recoverable"])
    exact = bool(recoverable and not wrong and not missed)

    noise_total = int(final_metrics["noise_pixels"])
    noise_attached = int(final_metrics["noise_pixels_in_symbol_groups"])
    noise_attachment_percent = 100.0 * noise_attached / max(1, noise_total)

    return {
        **equation_flags(idx, split, candidate["candidate_metrics"]),
        "source": source,
        "equation": equation,
        "candidate_recoverable": recoverable,
        "merger_correct": bool(not wrong and not missed),
        "upstream_correct": exact,
        "candidate_count": len(candidate["segments"]),
        "final_group_count": len(final_segments),
        "exact_merger": exact,
        "wrong_merge": bool(wrong),
        "missed_merge": bool(missed),
        "wrong_and_missed": bool(wrong and missed),
        "wrong_merge_groups": wrong_groups,
        "missed_symbol_instances": missed_instances,
        "extra_symbol_groups_from_missed_merges": extra_symbol_groups,
        "symbol_symbol_wrong_merge_groups": int(final_metrics["mixed_components_significant"]),
        "symbol_non_symbol_wrong_merge_groups": int(final_metrics["symbol_non_symbol_mixed_components_significant"]),
        "standalone_zero_owned_groups": int(final_metrics["standalone_zero_owned_components"]),
        "noise_pixels_total": noise_total,
        "noise_pixels_in_symbol_groups": noise_attached,
        "noise_attachment_percent": round(noise_attachment_percent, 4),
        "unassigned_pixels_in_symbol_groups": int(final_metrics["unassigned_pixels_in_symbol_groups"]),
        "symbol_foreign_pixels": int(final_metrics["foreign_pixels"]),
        "non_symbol_pixels_in_symbol_groups": int(final_metrics["non_symbol_pixels_in_symbol_groups"]),
    }


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter()
    for row in rows:
        c["candidate_recoverable"] += int(row["candidate_recoverable"])
        c["exact_merger"] += int(row["exact_merger"])
        c["wrong_merge"] += int(row["segmentation_recoverable"] and row["wrong_merge"])
        c["missed_merge"] += int(row["segmentation_recoverable"] and row["missed_merge"])
        c["wrong_and_missed"] += int(row["segmentation_recoverable"] and row["wrong_and_missed"])
        c["wrong_merge_groups"] += int(row["wrong_merge_groups"]) if row["segmentation_recoverable"] else 0
        c["missed_symbol_instances"] += int(row["missed_symbol_instances"]) if row["segmentation_recoverable"] else 0
        c["extra_symbol_groups"] += (
            int(row["extra_symbol_groups_from_missed_merges"]) if row["segmentation_recoverable"] else 0
        )
        c["candidate_count"] += int(row["candidate_count"])
        c["final_group_count"] += int(row["final_group_count"])
        c["symbol_symbol_wrong_merge_groups"] += int(row["symbol_symbol_wrong_merge_groups"])
        c["symbol_non_symbol_wrong_merge_groups"] += int(row["symbol_non_symbol_wrong_merge_groups"])
        c["standalone_zero_owned_groups"] += int(row["standalone_zero_owned_groups"])
        c["noise_pixels_total"] += int(row["noise_pixels_total"])
        c["noise_pixels_in_symbol_groups"] += int(row["noise_pixels_in_symbol_groups"])
        c["unassigned_pixels_in_symbol_groups"] += int(row["unassigned_pixels_in_symbol_groups"])

    ceiling = c["candidate_recoverable"]
    exact = c["exact_merger"]
    merger_errors = ceiling - exact
    only_wrong = c["wrong_merge"] - c["wrong_and_missed"]
    only_missed = c["missed_merge"] - c["wrong_and_missed"]

    return {
        "total_equations": n,
        "recoverable_equations": ceiling,
        "unrecoverable_equations": n - ceiling,
        "correctly_merged_equations": exact,
        "incorrectly_merged_equations": n - exact,
        "equations_with_incorrect_merging": sum(not row["merger_correct"] for row in rows),
        "upstream_correct_equations": exact,
        "overall_structural_accuracy": exact / n if n else None,
        "overall_structural_accuracy_percent": round(100 * exact / n, 3) if n else None,
        "conditional_merger_accuracy": exact / ceiling if ceiling else None,
        "conditional_merger_accuracy_percent": round(100 * exact / ceiling, 3) if ceiling else None,
        "merger_error_equations": merger_errors,
        "wrong_merge_equations": c["wrong_merge"],
        "missed_merge_equations": c["missed_merge"],
        "wrong_and_missed_equations": c["wrong_and_missed"],
        "wrong_merge_only_equations": only_wrong,
        "missed_merge_only_equations": only_missed,
        "wrong_merge_groups": c["wrong_merge_groups"],
        "missed_symbol_instances": c["missed_symbol_instances"],
        "extra_symbol_groups_from_missed_merges": c["extra_symbol_groups"],
        "symbol_symbol_wrong_merge_groups": c["symbol_symbol_wrong_merge_groups"],
        "symbol_non_symbol_wrong_merge_groups": c["symbol_non_symbol_wrong_merge_groups"],
        "standalone_zero_owned_groups": c["standalone_zero_owned_groups"],
        "noise_pixels_total": c["noise_pixels_total"],
        "noise_pixels_in_symbol_groups": c["noise_pixels_in_symbol_groups"],
        "noise_attachment_percent": round(
            100 * c["noise_pixels_in_symbol_groups"] / max(1, c["noise_pixels_total"]), 4
        ),
        "unassigned_pixels_in_symbol_groups": c["unassigned_pixels_in_symbol_groups"],
        "mean_candidate_count": round(c["candidate_count"] / n, 3) if n else None,
        "mean_final_group_count": round(c["final_group_count"] / n, 3) if n else None,
    }


def evaluate(results, threshold: float) -> tuple[dict, list[dict]]:
    """Consume streamed equation inputs and groups; return summary and CSV rows."""
    rows = [evaluate_equation(**item) for item in results]
    if not rows:
        raise ValueError("No equations evaluated")
    summary = summarize_rows(rows)
    summary.update(
        {
            "selected_threshold": threshold,
            "tolerances": dict(TOLERANCE),
            "metric_definitions": {
                "overall_structural_accuracy": "Correctly merged equations / all equations; unrecoverable segmentation is incorrect.",
                "conditional_merger_accuracy": "Correctly merged equations / recoverable equations; null when none are recoverable.",
                "merger_errors": "Wrong/missed errors are judged on candidate-recoverable equations, as in the original evaluation.",
                "standalone_noise": "Standalone zero-owned groups are allowed; significant symbol/noise mixing is a wrong merge.",
            },
        }
    )
    return summary, rows


def save_results(directory: Path, summary: dict, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (directory / "test_equations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
