from __future__ import annotations

import argparse
import json

from ml_segmentation.data_pipeline.preprocessing import stream_samples
from ml_segmentation.segmentation.segmentation import SEGMENTATION_METHODS
from ml_segmentation.merger import merger as merging
from ml_segmentation.merger.evaluate_merger import evaluate, evaluate_equation, save_results, summarize_rows


def select_threshold(model, kind: str, raster: int, segmentation: dict, method: str) -> dict:
    """Search only validation equations, scoring each candidate pair once."""
    rows_by_threshold = {threshold: [] for threshold in merging.THRESHOLDS}
    merge = merging.MERGER_METHODS[method]
    npz = merging.DATA_ROOT / "validation" / "preprocessed.npz"
    for idx, image, gt, noise, equation, annotations, source in stream_samples(npz):
        candidate = merging.generate_candidates(image, gt, noise, annotations, segmentation, merging.TOLERANCE)
        segments = candidate["segments"]
        scores = merging.score_pairs(model, kind, raster, segments, image.shape, merging.DEVICE)
        for threshold in merging.THRESHOLDS:
            groups = merge(model, segments, image.shape, threshold, merging.DEVICE, raster, scores=scores)
            rows_by_threshold[threshold].append(
                evaluate_equation(
                    idx=idx,
                    source=source,
                    equation=equation,
                    image=image,
                    gt=gt,
                    noise=noise,
                    annotations=annotations,
                    candidate=candidate,
                    groups=groups,
                    split="validation",
                )
            )
        if (idx + 1) % 500 == 0:
            print(f"Validation: {idx + 1:,} equations", flush=True)
    if not rows_by_threshold[merging.THRESHOLDS[0]]:
        raise ValueError("Validation split has no equations")
    sweep = [{"threshold": threshold, **summarize_rows(rows)} for threshold, rows in rows_by_threshold.items()]
    best = max(
        sweep,
        key=lambda row: (
            row["correctly_merged_equations"],
            -row["wrong_merge_equations"],
            -row["missed_merge_equations"],
            row["threshold"],
        ),
    )
    boundary = best["threshold"] in (merging.THRESHOLDS[0], merging.THRESHOLDS[-1])
    print(
        f"Validation selected {best['threshold']:.2f} "
        f"({best['overall_structural_accuracy_percent']:.2f}% structural accuracy)"
        + ("; optimum is at the search boundary" if boundary else ""),
        flush=True,
    )
    return {
        "split": "validation",
        "selected_threshold": best["threshold"],
        "threshold_search": list(merging.THRESHOLDS),
        "best_at_boundary": boundary,
        "best": best,
        "threshold_sweep": sweep,
    }


def run(segmentation: str, merger: str, split: str) -> dict:
    settings = {"method": segmentation, **merging.SEGMENTATION_PARAMETERS}
    output = merging.OUTPUT_ROOT / merger
    checkpoint = output / "model.pt"
    config_path = output / "config.json"
    expected_kind = "geometry" if merger == "geometry_mlp_simple" else "cnn"

    if split == "all":
        print(f"Generating {segmentation} pairs for {merger}", flush=True)
        training = merging.generate_pairs(
            merging.DATA_ROOT / "train" / "preprocessed.npz",
            settings,
            expected_kind,
            merging.MAX_TRAIN_PAIRS,
            merging.TRAIN_SEED,
        )
        validation = merging.generate_pairs(
            merging.DATA_ROOT / "validation" / "preprocessed.npz",
            settings,
            expected_kind,
            merging.MAX_VALIDATION_PAIRS,
            merging.VALIDATION_SEED,
        )
        output.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "evaluation.json", "test_equations.csv"):
            (output / name).unlink(missing_ok=True)
        merging.train_model(expected_kind, training, validation, checkpoint, settings, merger)
        pair_summary = {"train": training["summary"], "validation": validation["summary"]}
        del training, validation
        model, kind, raster, saved = merging.load_affinity(checkpoint, merging.DEVICE, settings)
        selection = select_threshold(model, kind, raster, settings, merger)
        config = {
            "merger": merger,
            "segmentation": settings,
            "affinity_model": kind,
            "selected_threshold": selection["selected_threshold"],
            "validation": selection,
            "training": {
                "device": merging.DEVICE,
                "max_epochs": merging.MAX_EPOCHS,
                "batch_size": merging.BATCH_SIZE,
                "learning_rate": merging.LEARNING_RATE,
                "early_stopping_patience": merging.EARLY_STOPPING_PATIENCE,
                "early_stopping_min_delta": merging.EARLY_STOPPING_MIN_DELTA,
                "train_pair_limit": merging.MAX_TRAIN_PAIRS,
                "validation_pair_limit": merging.MAX_VALIDATION_PAIRS,
                "negative_ratio": merging.NEGATIVE_RATIO,
                "train_seed": merging.TRAIN_SEED,
                "validation_seed": merging.VALIDATION_SEED,
                "raster_size": raster,
                "best_epoch": saved["best_epoch"],
                "validation_loss": saved["validation_loss"],
                "pairs": pair_summary,
            },
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing {checkpoint} or {config_path}; run --split all first")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("merger") != merger or config.get("segmentation") != settings:
        raise ValueError("Saved configuration does not match the selected merger/segmentation; run --split all")
    threshold = config.get("selected_threshold")
    if threshold not in merging.THRESHOLDS:
        raise ValueError("Saved validation threshold must be between 0.70 and 1.00 at 0.01 intervals")
    model, kind, raster, saved = merging.load_affinity(checkpoint, merging.DEVICE, settings)
    if kind != expected_kind or saved.get("merger") != merger:
        raise ValueError("Saved checkpoint does not match the selected merger")
    merge = merging.MERGER_METHODS[merger]

    def test_results():
        npz = merging.DATA_ROOT / "test" / "preprocessed.npz"
        for idx, image, gt, noise, equation, annotations, source in stream_samples(npz):
            candidate = merging.generate_candidates(image, gt, noise, annotations, settings, merging.TOLERANCE)
            groups = merge(model, candidate["segments"], image.shape, threshold, merging.DEVICE, raster)
            yield dict(
                idx=idx,
                source=source,
                equation=equation,
                image=image,
                gt=gt,
                noise=noise,
                annotations=annotations,
                candidate=candidate,
                groups=groups,
            )

    summary, rows = evaluate(test_results(), threshold)
    summary.update({"merger": merger, "segmentation": settings, "split": "test"})
    save_results(output, summary, rows)
    conditional = summary["conditional_merger_accuracy_percent"]
    conditional_text = "unavailable" if conditional is None else f"{conditional:.2f}%"
    print(
        f"{merger} / {segmentation}: {summary['correctly_merged_equations']:,}/"
        f"{summary['total_equations']:,} correct | overall {summary['overall_structural_accuracy_percent']:.2f}%"
        f" | recoverable-only {conditional_text} | threshold {threshold:.2f}",
        flush=True,
    )
    print(f"Saved {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", choices=SEGMENTATION_METHODS, default="baseline")
    parser.add_argument("--merger", choices=merging.MERGER_METHODS, default="geometry_mlp_simple")
    parser.add_argument("--split", choices=("all", "test"), default="test")
    args = parser.parse_args()
    run(args.segmentation, args.merger, args.split)


if __name__ == "__main__":
    main()
