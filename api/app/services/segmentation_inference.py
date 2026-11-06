from functools import lru_cache
import json

import numpy as np
from fastapi import HTTPException

from ml_segmentation.classification import model, post_classification as post
from ml_segmentation.data_pipeline import preprocessing
from ml_segmentation.merger import merger as merging
from ml_segmentation.segmentation.segmentation import SEGMENTATION_METHODS

SEGMENTATION_MODELS = {
    "ccl_geometry_mlp_resnet": "baseline",
    "ccl_stroke_width_geometry_mlp_resnet": "stroke_width",
}
MERGER_METHOD = "geometry_mlp_simple"
CLASSIFIER = "resnet18"


def preprocess_segmentation_image(image: np.ndarray) -> np.ndarray:
    """Use the experiment's image rules, including its variable-width canvas."""
    gray = preprocessing.image_to_grayscale(image)
    cleaned = preprocessing.remove_dark_borders(gray)
    binary = preprocessing.binarize_image(cleaned)
    binary = preprocessing.remove_isolated_noise(binary)
    return preprocessing.resize_and_pad(preprocessing.crop_to_ink(binary))


@lru_cache(maxsize=1)
def load_pipeline() -> dict:
    """Load pretrained segmentation, merging, and classification models."""
    merger_dir = merging.OUTPUT_ROOT / MERGER_METHOD
    classifier_dir = model.OUTPUT_ROOT / CLASSIFIER

    segmentation = {"method": "baseline", **merging.SEGMENTATION_PARAMETERS}

    merger_config = json.loads((merger_dir / "config.json").read_text())
    classifier_config = json.loads((classifier_dir / "config.json").read_text())

    affinity, _, raster, _ = merging.load_affinity(merger_dir / "model.pt", model.DEVICE, segmentation)

    classifier, _ = model.load_classifier(classifier_dir / "model.pt", CLASSIFIER)

    return {
        "affinity": affinity,
        "raster": raster,
        "threshold": merger_config["selected_threshold"],
        "segmentation": segmentation,
        "classifier": classifier,
        "operator": post.load_operator(classifier_dir / "operator_mlp.pt", CLASSIFIER),
        "post_settings": classifier_config["post_classification"],
    }


def recognize_segmentation(processed_image: np.ndarray, model_name: str) -> str:
    """Run CCL, optional stroke-width cuts, shared merging, and final classification."""
    pipeline = load_pipeline()
    settings = pipeline["segmentation"]
    segments, _ = SEGMENTATION_METHODS[SEGMENTATION_MODELS[model_name]](
        processed_image, settings["ratio"], settings["min_side_fraction"], settings["max_cuts"]
    )
    if not segments:
        return ""
    groups = merging.MERGER_METHODS[MERGER_METHOD](
        pipeline["affinity"], segments, processed_image.shape, pipeline["threshold"], model.DEVICE, pipeline["raster"]
    )
    merged = merging.groups_to_segments(segments, groups)
    data = model.predict_segments(merged, processed_image.shape, pipeline["classifier"])
    stages = post.apply_methods(data, ["all"], pipeline["post_settings"], pipeline["operator"])
    predictions = stages[next(reversed(stages))]
    return "".join(model.CLASSES[int(label)] for label in predictions if model.CLASSES[int(label)] != "NOISE")
