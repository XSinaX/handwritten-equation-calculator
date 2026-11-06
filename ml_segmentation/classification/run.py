"""Train and evaluate classifiers, or evaluate a frozen complete pipeline."""

from __future__ import annotations

import argparse
import json

from ml_segmentation.classification import model, post_classification as post
from ml_segmentation.classification.evaluate_classification import evaluate, save_results
from ml_segmentation.merger import merger as merging
from ml_segmentation.segmentation.segmentation import SEGMENTATION_METHODS


def run(segmentation: str, merger: str, name: str, post_methods: list[str], split: str) -> dict:
    if split not in ("all", "test"):
        raise ValueError("split must be 'all' or 'test'")
    methods = post.selected_methods(post_methods)
    settings = {"method": segmentation, **merging.SEGMENTATION_PARAMETERS}
    merger_output = merging.OUTPUT_ROOT / merger
    merger_config = merger_output / "config.json"
    if not merger_config.is_file() or not (merger_output / "model.pt").is_file():
        raise FileNotFoundError(f"Missing merger model/config in {merger_output}; train the merger separately first")
    saved_merger = json.loads(merger_config.read_text(encoding="utf-8"))
    if saved_merger.get("segmentation") != settings or saved_merger.get("merger") != merger:
        raise ValueError("Saved merger is incompatible with the selected segmentation/method")
    threshold = saved_merger.get("selected_threshold")
    if threshold not in merging.THRESHOLDS:
        raise ValueError("Missing or invalid validation-selected merger threshold")
    merger_model, _, raster, merger_checkpoint = merging.load_affinity(
        merger_output / "model.pt", model.DEVICE, settings
    )
    if merger_checkpoint.get("merger") != merger:
        raise ValueError("Merger checkpoint does not match the selected method")

    output = model.OUTPUT_ROOT / name
    config_path = output / "config.json"
    data_by_split = {}
    learned = None
    if split == "all":
        training = model.prepare_training_data("train", settings)
        validation = model.prepare_training_data("validation", settings)
        output.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.json",
            "operator_mlp.pt",
            "operator_training_history.csv",
            "train.json",
            "validation.json",
            "ml_segmentation.json",
            "train_equations.csv",
            "validation_equations.csv",
            "test_equations.csv",
        ):
            (output / filename).unlink(missing_ok=True)
        training_config = model.train_classifier(name, training, validation, output, settings)
        del training, validation
        classifier, _ = model.load_classifier(output / "model.pt", name)
        for dataset_split in ("train", "validation"):
            data_by_split[dataset_split] = model.collect_split(
                dataset_split, settings, merger, merger_model, raster, threshold, classifier
            )
        post_settings = {"syntax": dict(post.SYNTAX_SETTINGS)}
        if "operator_mlp" in methods:
            learned = post.train_operator(data_by_split["train"], data_by_split["validation"], name, output)
            post_settings["operator_mlp"] = post.tune_operator(data_by_split["validation"], learned)
        config = {
            "model": name,
            "image_size": model.IMAGE_SIZE,
            "training": training_config,
            "training_segmentation": settings,
            "post_classification": post_settings,
            "post_training_merger": merger,
            "post_training_merge_threshold": threshold,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing classifier configuration {config_path}; run --split all first")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model") != name or config.get("image_size") != model.IMAGE_SIZE:
        raise ValueError("Classifier configuration does not match the selected model/crop size")
    classifier, _ = model.load_classifier(output / "model.pt", name)
    post_settings = config.get("post_classification", {})
    if "syntax" in methods and "syntax" not in post_settings:
        raise ValueError("Missing saved syntax settings")
    if "operator_mlp" in methods:
        if "operator_mlp" not in post_settings:
            raise ValueError(
                "Missing validation-selected operator settings; train with --post operator_mlp --split all"
            )
        learned = post.load_operator(output / "operator_mlp.pt", name)

    reports = {}
    for dataset_split in ("train", "validation", "test") if split == "all" else ("test",):
        data = data_by_split.pop(dataset_split, None)
        if data is None:
            data = model.collect_split(dataset_split, settings, merger, merger_model, raster, threshold, classifier)
        stages = post.apply_methods(data, methods, post_settings, learned)
        summary, rows = evaluate(data, stages)
        summary.update(
            {
                "model": name,
                "segmentation": settings,
                "merger": merger,
                "merge_threshold": threshold,
                "post_classification_methods": methods,
                "split": dataset_split,
            }
        )
        save_results(output, dataset_split, summary, rows)
        reports[dataset_split] = summary
        base = summary["base"]["overall"]
        final = summary["final"]["overall"]
        conditional = summary["final"]["conditional"]["exact_sequence_accuracy_percent"]
        conditional_text = "unavailable" if conditional is None else f"{conditional:.2f}%"
        print(
            f"{name} / {dataset_split}: {summary['total_equations']:,} equations | "
            f"base {base['exact_sequence_accuracy_percent']:.2f}% -> final {final['exact_sequence_accuracy_percent']:.2f}% | "
            f"TER {final['token_error_rate_percent']:.3f}% | upstream-correct accuracy {conditional_text}",
            flush=True,
        )
    print(f"Saved {output}", flush=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", choices=SEGMENTATION_METHODS, default="baseline")
    parser.add_argument("--merger", choices=merging.MERGER_METHODS, default="geometry_mlp_simple")
    parser.add_argument("--model", choices=model.CLASSIFICATION_METHODS, default="resnet18")
    parser.add_argument(
        "--post", nargs="+", choices=("all", "none", *post.POST_CLASSIFICATION_METHODS), default=["all"]
    )
    parser.add_argument("--split", choices=("all", "test"), default="test")
    args = parser.parse_args()
    try:
        post.selected_methods(args.post)
    except ValueError as error:
        parser.error(str(error))
    run(args.segmentation, args.merger, args.model, args.post, args.split)


if __name__ == "__main__":
    main()
