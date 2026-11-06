from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ml_segmentation.classification.model import CLASSES


def align_tokens(reference: list[str], predicted: list[str], *, include_alignment: bool = False) -> dict:
    """Levenshtein alignment, with priority match > substitution > deletion > insertion."""
    n, m = len(reference), len(predicted)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    ties = [[False] * (m + 1) for _ in range(n + 1)] if include_alignment else None
    for i in range(1, n + 1):
        dp[i][0] = i
        trace[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        trace[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = reference[i - 1] == predicted[j - 1]
            options = [
                (dp[i - 1][j - 1] + (0 if match else 1), 0, "M" if match else "S"),
                (dp[i - 1][j] + 1, 1, "D"),
                (dp[i][j - 1] + 1, 2, "I"),
            ]
            cost, _, op = min(options)
            dp[i][j], trace[i][j] = cost, op
            if ties is not None:
                ties[i][j] = sum(option[0] == cost for option in options) > 1
    i, j = n, m
    counts: Counter[str] = Counter()
    alignment = []
    while i or j:
        op = trace[i][j]
        counts[op] += 1
        if include_alignment:
            ri = i - 1 if op != "I" else None
            pi = j - 1 if op != "D" else None
            alignment.append(
                {
                    "operation": op,
                    "reference_index": ri,
                    "predicted_index": pi,
                    "reference": reference[ri] if ri is not None else None,
                    "predicted": predicted[pi] if pi is not None else None,
                    "ambiguous": ties[i][j],
                }
            )
        if op in ("M", "S"):
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            j -= 1
        else:
            raise AssertionError(f"Invalid alignment at ({i}, {j}): {op!r}")
    result = {
        "edit_distance": int(dp[n][m]),
        "matched_tokens": int(counts["M"]),
        "substitutions": int(counts["S"]),
        "deletions": int(counts["D"]),
        "insertions": int(counts["I"]),
    }
    if include_alignment:
        result["alignment"] = list(reversed(alignment))
        result["alignment_ambiguous"] = any(step["ambiguous"] for step in alignment)
    return result


def prediction_rows(data: dict, predictions) -> list[dict]:
    """Score sequences independently of structural correctness."""
    rows = []
    for i, equation in enumerate(data["equations"]):
        lo, hi = int(data["offsets"][i]), int(data["offsets"][i + 1])
        predicted = [CLASSES[int(label)] for label in predictions[lo:hi] if CLASSES[int(label)] != "NOISE"]
        reference = equation["reference_tokens"]
        edits = align_tokens(reference, predicted)
        rows.append(
            {
                "prediction": "".join(predicted),
                "sequence_correct": reference == predicted,
                "ground_truth_tokens": len(reference),
                "token_edit_distance": edits["edit_distance"],
                "substitutions": edits["substitutions"],
                "insertions": edits["insertions"],
                "deletions": edits["deletions"],
                "upstream_correct": bool(equation["segmentation_recoverable"] and equation["merger_correct"]),
            }
        )
    return rows


def sequence_metrics(rows: list[dict]) -> dict:
    count = len(rows)
    correct = sum(row["sequence_correct"] for row in rows)
    tokens = sum(row["ground_truth_tokens"] for row in rows)
    distance = sum(row["token_edit_distance"] for row in rows)
    return {
        "equations": count,
        "correct_sequences": correct,
        "ground_truth_tokens": tokens,
        "exact_sequence_accuracy": correct / count if count else None,
        "exact_sequence_accuracy_percent": round(100 * correct / count, 4) if count else None,
        "token_edit_distance": distance,
        "token_error_rate": distance / tokens if tokens else None,
        "token_error_rate_percent": round(100 * distance / tokens, 4) if tokens else None,
        "substitutions": sum(row["substitutions"] for row in rows),
        "insertions": sum(row["insertions"] for row in rows),
        "deletions": sum(row["deletions"] for row in rows),
    }


def evaluate(data: dict, stages: dict) -> tuple[dict, list[dict]]:
    if not data["equations"]:
        raise ValueError("No equations evaluated")
    stage_rows = {name: prediction_rows(data, prediction) for name, prediction in stages.items()}
    metrics = {
        name: {
            "overall": sequence_metrics(rows),
            "conditional": sequence_metrics([row for row in rows if row["upstream_correct"]]),
        }
        for name, rows in stage_rows.items()
    }
    first, last = stage_rows["base"], stage_rows[next(reversed(stage_rows))]
    n = len(first)
    changes = {}
    previous = first
    for name, rows in stage_rows.items():
        if name == "base":
            continue
        gained = sum(not a["sequence_correct"] and b["sequence_correct"] for a, b in zip(previous, rows))
        lost = sum(a["sequence_correct"] and not b["sequence_correct"] for a, b in zip(previous, rows))
        conditional_gained = sum(
            not a["sequence_correct"] and b["sequence_correct"] and b["upstream_correct"]
            for a, b in zip(previous, rows)
        )
        conditional_lost = sum(
            a["sequence_correct"] and not b["sequence_correct"] and b["upstream_correct"]
            for a, b in zip(previous, rows)
        )
        eligible = sum(row["upstream_correct"] for row in rows)
        changes[name] = {
            "corrected_equations": gained,
            "previously_correct_made_incorrect": lost,
            "net_correct_equations": gained - lost,
            "net_accuracy_percentage_points": round(100 * (gained - lost) / n, 4),
            "conditional_corrected_equations": conditional_gained,
            "conditional_previously_correct_made_incorrect": conditional_lost,
            "conditional_net_accuracy_percentage_points": (
                round(100 * (conditional_gained - conditional_lost) / eligible, 4) if eligible else None
            ),
        }
        previous = rows
    counts = Counter(
        {
            name: 0
            for name in (
                "segmentation_unrecoverable",
                "merger_incorrect",
                "classification_incorrect",
                "all_stages_correct",
            )
        }
    )
    result_rows = []
    for i, equation in enumerate(data["equations"]):
        recoverable = bool(equation["segmentation_recoverable"])
        merger_correct = bool(equation["merger_correct"])
        upstream = bool(recoverable and merger_correct)
        base, final = first[i], last[i]
        category = (
            "segmentation_unrecoverable"
            if not recoverable
            else (
                "merger_incorrect"
                if not merger_correct
                else "classification_incorrect" if not final["sequence_correct"] else "all_stages_correct"
            )
        )
        counts[category] += 1
        row = {
            "equation_id": equation["equation_id"],
            "sample_index": equation["sample_index"],
            "split": equation["split"],
            "source": equation["source"],
            "ground_truth": equation["equation"],
            "base_prediction": base["prediction"],
            "final_prediction": final["prediction"],
            "segmentation_recoverable": recoverable,
            "segmentation_unrecoverable": not recoverable,
            "wrong_merge": bool(equation["wrong_merge"]),
            "missed_merge": bool(equation["missed_merge"]),
            "merger_correct": merger_correct,
            "upstream_correct": upstream,
            "base_sequence_correct": base["sequence_correct"],
            "final_sequence_correct": final["sequence_correct"],
            "classification_correct_given_upstream": bool(upstream and final["sequence_correct"]),
            "ground_truth_tokens": base["ground_truth_tokens"],
            "base_token_edit_distance": base["token_edit_distance"],
            "final_token_edit_distance": final["token_edit_distance"],
            "stage_attribution": category,
        }
        for name, rows in stage_rows.items():
            if name != "base":
                row[name + "_prediction"] = rows[i]["prediction"]
                row[name + "_sequence_correct"] = rows[i]["sequence_correct"]
                row[name + "_token_edit_distance"] = rows[i]["token_edit_distance"]
        result_rows.append(row)
    assert sum(counts.values()) == n
    summary = {
        "total_equations": n,
        "base": metrics["base"],
        "final": metrics[next(reversed(metrics))],
        "stages": metrics,
        "post_classification_improvements": changes,
        "upstream_statistics": {
            "segmentation_recoverable_equations": sum(r["segmentation_recoverable"] for r in result_rows),
            "segmentation_unrecoverable_equations": sum(r["segmentation_unrecoverable"] for r in result_rows),
            "incorrect_merging_equations": sum(not r["merger_correct"] for r in result_rows),
            "correct_upstream_equations": sum(r["upstream_correct"] for r in result_rows),
            "incorrect_final_with_correct_upstream": sum(
                r["upstream_correct"] and not r["final_sequence_correct"] for r in result_rows
            ),
            "incorrect_final_with_incorrect_upstream": sum(
                not r["upstream_correct"] and not r["final_sequence_correct"] for r in result_rows
            ),
            "correct_final_despite_imperfect_merging": sum(
                not r["merger_correct"] and r["final_sequence_correct"] for r in result_rows
            ),
        },
        "stage_attribution": dict(counts),
        "metric_definitions": {
            "conditional_population": "Segmentation recoverable AND merger structurally correct.",
            "TER_denominator": "Sum of ground-truth symbol tokens in the relevant population; never equation count.",
            "stage_attribution": "Earliest failing stage, independent of whether an imperfect upstream pipeline happened to predict the correct sequence.",
            "classification_correct_given_upstream": "True only when upstream is correct and final sequence is exact.",
        },
    }
    return summary, result_rows


def save_results(output: Path, split: str, summary: dict, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / f"{split}_equations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
