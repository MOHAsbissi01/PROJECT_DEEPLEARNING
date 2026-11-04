"""Training utilities: compilation, fitting with checkpoints, run bookkeeping."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from . import config
from .models import LOSSES, masked_reconstruction_loss


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def compile_segmentation(model, learning_rate=1e-3, loss="bce", monitor_threshold=0.5):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=LOSSES[loss],
        metrics=[
            keras.metrics.Precision(name="precision", thresholds=monitor_threshold),
            keras.metrics.Recall(name="recall", thresholds=monitor_threshold),
            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=monitor_threshold, name="iou"),
        ],
    )
    return model


def compile_autoencoder(model, learning_rate=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=masked_reconstruction_loss)
    return model


def fit_model(model, train_ds, val_ds, run_dir: Path, run_config: dict, max_epochs=50, patience=7, initial_epoch=0, verbose=1):
    """Train with checkpointing on val_loss + early stopping, and save the run artefacts."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.keras"
    write_json(run_dir / "run_config.json", run_config)

    callbacks = [
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv"), append=initial_epoch > 0),
        keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=initial_epoch + max_epochs, initial_epoch=initial_epoch,
        callbacks=callbacks, verbose=verbose,
    )
    duration = time.perf_counter() - start

    values = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    val_losses = np.array(values["val_loss"])
    if not np.all(np.isfinite(val_losses)):
        raise ValueError("Non-finite validation loss: inspect the training log.")
    summary = {
        "epochs_completed": len(values["loss"]),
        "best_epoch": int(np.argmin(val_losses)) + 1 + initial_epoch,
        "best_validation_loss": float(val_losses.min()),
        "training_duration_seconds": duration,
        "devices": [d.name for d in tf.config.list_physical_devices("GPU")] or ["CPU"],
    }
    write_json(run_dir / "history.json", values)
    write_json(run_dir / "training_summary.json", summary)
    return {"history": values, "summary": summary, "checkpoint": checkpoint}


def merge_histories(first: dict, second: dict) -> dict:
    return {k: first.get(k, []) + second.get(k, []) for k in set(first) | set(second)}


def plot_history(history: dict, title: str, best_epoch: int | None = None, path: Path | None = None, metrics=None):
    import matplotlib.pyplot as plt

    metrics = metrics or [m for m in ("loss", "precision", "recall", "iou") if m in history]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.6), squeeze=False)
    epochs = np.arange(1, len(history["loss"]) + 1)
    for axis, metric in zip(axes[0], metrics):
        axis.plot(epochs, history[metric], label="train")
        if f"val_{metric}" in history:
            axis.plot(epochs, history[f"val_{metric}"], label="validation")
        if best_epoch:
            axis.axvline(best_epoch, color="gray", linestyle="--", label="best val_loss")
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        if metric != "loss":
            axis.set_ylim(0, 1)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def background_error_scale(autoencoder, records, percentile=99.0, smooth=9, height=config.IMG_HEIGHT, width=config.IMG_WIDTH):
    """High percentile of the smoothed reconstruction error on *background* training pixels."""
    from .data import load_image_mask

    values = []
    for record in records:
        image, mask = load_image_mask(record, height, width)
        reconstruction = autoencoder(image[np.newaxis], training=False)
        error = tf.reduce_mean(tf.square(image[np.newaxis] - reconstruction), axis=-1, keepdims=True)
        error = tf.nn.avg_pool2d(error, smooth, 1, "SAME")[0, ..., 0].numpy()
        values.append(error[mask[..., 0] == 0])
    return float(np.percentile(np.concatenate(values), percentile))
