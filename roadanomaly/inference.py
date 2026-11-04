"""Inference and experiment-loading helpers (used by the notebook and Streamlit).

An *experiment* is a model-selection folder containing ``validation_selection.json``,
the copied checkpoints in ``models/<name>.keras``, the run configurations and,
after the final evaluation, ``final_evaluation/test_results.json``.  Experiments
produced by the original Colab notebook keep working unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .data import prepare_image


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    selection_path: Path
    payload: dict
    run_configs: dict = field(default_factory=dict)
    test_results: dict | None = None

    @property
    def directory(self) -> Path:
        return self.selection_path.parent

    @property
    def run_id(self) -> str:
        return self.directory.name

    @property
    def model_names(self) -> list[str]:
        return list(self.payload["models"])

    @property
    def primary_model(self) -> str:
        return self.payload.get("primary_model", self.model_names[0])

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.payload["image_height"]), int(self.payload["image_width"])

    def threshold(self, model_name: str) -> float:
        return float(self.payload["models"][model_name]["threshold"])

    def checkpoint(self, model_name: str) -> Path:
        local = self.directory / "models" / f"{model_name}.keras"
        if local.is_file():
            return local
        relative = self.payload["models"][model_name].get("checkpoint_relative_path", "")
        # Colab paths were relative to the project root, i.e. 3 levels above the selection dir.
        for root in self.directory.parents[:4]:
            candidate = root / relative
            if relative and candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Checkpoint not found for {model_name} in {self.directory}")

    @property
    def splits(self) -> dict | None:
        for cfg in self.run_configs.values():
            if "splits" in cfg:
                return cfg["splits"]
        return None

    def label(self, project_root: Path = config.PROJECT_ROOT) -> str:
        try:
            where = self.selection_path.parent.relative_to(project_root)
        except ValueError:
            where = self.selection_path.parent
        suffix = " + test" if self.test_results else ""
        return f"{self.run_id}  [{', '.join(self.model_names)}{suffix}]  ({where.parts[0] if where.parts else where})"


def discover_experiments(root: Path = config.PROJECT_ROOT, max_depth: int = 6) -> list[Path]:
    """All validation_selection.json files under ``root`` (most complete/recent first)."""
    found = []
    root = Path(root)
    for path in root.rglob("validation_selection.json"):
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        hidden = any(p.startswith(".") for p in path.relative_to(root).parts)
        has_models = (path.parent / "models").is_dir()  # skips copies kept as report evidence
        if depth <= max_depth and not hidden and has_models:
            found.append(path)

    def sort_key(path: Path):
        has_test = (path.parent / "final_evaluation" / "test_results.json").is_file()
        has_analysis = (path.parent / "final_evaluation" / "analysis").is_dir()
        n_models = len(json.loads(path.read_text(encoding="utf-8")).get("models", {}))
        return (has_test, n_models, has_analysis, path.parent.name, path.stat().st_mtime)

    return sorted(found, key=sort_key, reverse=True)


def load_experiment(selection_path: Path) -> Experiment:
    selection_path = Path(selection_path)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    run_configs = {}
    for name in payload.get("models", {}):
        cfg = selection_path.parent / f"{name}_run_config.json"
        if cfg.is_file():
            run_configs[name] = json.loads(cfg.read_text(encoding="utf-8"))
    test_path = selection_path.parent / "final_evaluation" / "test_results.json"
    test_results = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
    return Experiment(selection_path, payload, run_configs, test_results)


def load_model(path: Path):
    import keras

    from . import models  # noqa: F401  (registers the custom losses)

    return keras.models.load_model(path, compile=False)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_probabilities(model, rgb: np.ndarray, image_size=(config.IMG_HEIGHT, config.IMG_WIDTH), native: bool = True) -> np.ndarray:
    """Anomaly score map in [0, 1]; resized back (bilinear) to the input resolution if ``native``."""
    import tensorflow as tf

    height, width = image_size
    prepared = prepare_image(rgb, height, width)
    scores = model(prepared[np.newaxis], training=False)[0]
    if native and tuple(scores.shape[:2]) != tuple(rgb.shape[:2]):
        scores = tf.image.resize(scores, rgb.shape[:2], method="bilinear")
    scores = np.clip(np.asarray(scores)[..., 0], 0.0, 1.0)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite scores predicted.")
    return scores.astype(np.float32)


def autoencoder_reconstruction(scorer, rgb: np.ndarray, image_size=(config.IMG_HEIGHT, config.IMG_WIDTH)) -> np.ndarray | None:
    """Reconstruction produced by the autoencoder inside a scorer model (None for other models)."""
    from .models import AUTOENCODER_LAYER_NAME

    try:
        autoencoder = scorer.get_layer(AUTOENCODER_LAYER_NAME)
    except ValueError:
        return None
    prepared = prepare_image(rgb, *image_size)
    reconstruction = np.asarray(autoencoder(prepared[np.newaxis], training=False))[0]
    return (np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualisation helpers (pure numpy -> usable in matplotlib and Streamlit)
# ---------------------------------------------------------------------------
TP_COLOR, FP_COLOR, FN_COLOR = (0, 180, 0), (255, 0, 0), (0, 100, 255)


def colorize(scores: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    import matplotlib

    return (matplotlib.colormaps[cmap](np.clip(scores, 0, 1))[..., :3] * 255).astype(np.uint8)


def blend(rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return ((1 - alpha) * rgb + alpha * overlay_rgb).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    mask = mask.astype(bool)
    out[mask] = ((1 - alpha) * rgb[mask] + alpha * np.array(color)).astype(np.uint8)
    return out


def error_map(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Green = true positive, red = false positive, blue = false negative, black = true negative."""
    out = np.zeros((*y_true.shape, 3), dtype=np.uint8)
    out[y_true & y_pred] = TP_COLOR
    out[~y_true & y_pred] = FP_COLOR
    out[y_true & ~y_pred] = FN_COLOR
    return out
