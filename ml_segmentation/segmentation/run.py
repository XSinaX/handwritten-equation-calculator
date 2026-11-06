from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ml_segmentation.data_pipeline.preprocessing import DATASETS_ROOT, stream_samples
from ml_segmentation.segmentation.evaluate_segmentation import evaluate
from ml_segmentation.segmentation.segmentation import SEGMENTATION_METHODS

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
SPLITTING_PARAMETERS = {
    "max_relative_ink": 0.35,
    "min_side_fraction": 0.18,
    "max_cuts_per_component": 1,
}


def run(method: str, split: str) -> dict[str, dict]:
    """Evaluate all dataset splits, or only test; segmentation has no training."""
    if split not in ("all", "test"):
        raise ValueError("split must be 'all' or 'test'")
    segment = SEGMENTATION_METHODS[method]
    splits = ("train", "validation", "test") if split == "all" else ("test",)
    reports = {}
    for dataset_split in splits:
        npz = DATASETS_ROOT / dataset_split / "preprocessed.npz"

        def results():
            for idx, image, gt, noise, equation, annotations, source in stream_samples(npz):
                segments, stats = segment(image, **SPLITTING_PARAMETERS)
                yield dict(
                    idx=idx,
                    split=dataset_split,
                    source=source,
                    equation=equation,
                    image=image,
                    gt=gt,
                    noise=noise,
                    annotations=annotations,
                    segments=segments,
                    stats=stats,
                )

        metrics, rows = evaluate(results())
        report = {
            "method": method,
            "split": dataset_split,
            "dataset": str(npz),
            "splitting_parameters": SPLITTING_PARAMETERS,
            **metrics,
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_ROOT / f"{method}_{dataset_split}.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"{method} / {dataset_split}: {report['total_equations']:,} equations | "
            f"{report['original_components']:,} CCL components -> "
            f"{report['final_candidates']:,} candidates | {report['cuts']:,} cuts"
        )
        print(
            f"Recoverable: {report['recoverable_equations']:,} ({report['recoverability_percent']:.2f}%) | "
            f"Unrecoverable: {report['unrecoverable_equations']:,} ({report['unrecoverability_percent']:.2f}%)"
        )
        print(f"Saved {output}")
        reports[dataset_split] = report
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=SEGMENTATION_METHODS, default="baseline")
    parser.add_argument("--split", choices=("all", "test"), default="test")
    args = parser.parse_args()
    run(args.method, args.split)


if __name__ == "__main__":
    main()
