"""Course-aligned experiments (chapter 3: optimisation, regularisation, hyper-parameter tuning).

Every study trains small variants of the baseline CNN on the **training** partition, stops them with
early stopping on the validation loss, and ranks them on the **validation** partition only
(pooled pixel F1 at its best threshold + AUROC).  The test set is never touched here.

Studies
-------
* ``optimizers``      SGD vs SGD + momentum vs RMSprop vs Adam            (slides: optimisation algorithms)
* ``regularization``  none / dropout / L2 / batch-norm / dropout + L2     (slides: regularisation, batch-norm)
* ``search``          random search, then a finer search around the best  (slides: coarse-to-fine search)

Results are cached as one JSON file per run, so an interrupted study resumes where it stopped:

    python -m roadanomaly.experiments --study all
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import config

EXPERIMENTS_DIR = config.OUTPUT_DIR / "experiments"
MAX_EPOCHS = 40
PATIENCE = 8
LOSS = "bce_dice"
BASE = {"optimizer": "adam", "learning_rate": 1e-3, "batch_size": config.BATCH_SIZE, "dropout": 0.0, "l2": 0.0, "batch_norm": False}

OPTIMIZER_STUDY = {
    "sgd": {"optimizer": "sgd", "learning_rate": 1e-2},
    "sgd_momentum": {"optimizer": "momentum", "learning_rate": 1e-2},
    "rmsprop": {"optimizer": "rmsprop", "learning_rate": 1e-3},
    "adam": {"optimizer": "adam", "learning_rate": 1e-3},
}
REGULARIZATION_STUDY = {
    "adam": {},  # reference, shared with the optimiser study
    "dropout_0.4": {"dropout": 0.4},
    "l2_1e-4": {"l2": 1e-4},
    "batch_norm": {"batch_norm": True},
    "dropout_0.4+l2_1e-4": {"dropout": 0.4, "l2": 1e-4},
}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def _validation_scores(model, val_records, size):
    from . import data, inference

    truths, scores = [], []
    for record in val_records:
        truths.append(data.load_native_mask(record["mask_path"]))
        scores.append(inference.predict_probabilities(model, data.load_rgb(record["image_path"]), size))
    return list(zip(truths, scores))


def run_variant(study: str, name: str, overrides: dict, train_records, val_records, force: bool = False) -> dict:
    """Train one baseline variant (or load its cached result) and score it on validation."""
    import keras

    from . import data, metrics, models, training

    run_dir = EXPERIMENTS_DIR / study / name
    result_path = run_dir / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    hp = {**BASE, **overrides}
    keras.utils.set_random_seed(config.SEED)
    model = models.build_baseline_cnn(batch_norm=hp["batch_norm"], dropout=hp["dropout"], l2=hp["l2"])
    training.compile_segmentation(model, hp["learning_rate"], LOSS, optimizer=hp["optimizer"])
    train_ds = data.make_dataset(train_records, training=True, augmentation=True, batch_size=hp["batch_size"])
    val_ds = data.make_dataset(val_records, batch_size=hp["batch_size"])
    start = time.perf_counter()
    fit = training.fit_model(
        model, train_ds, val_ds, run_dir, {"study": study, "name": name, **hp, "loss": LOSS, "seed": config.SEED},
        max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=0,
    )
    pairs = _validation_scores(model, val_records, (config.IMG_HEIGHT, config.IMG_WIDTH))  # best weights restored
    sweep = metrics.threshold_sweep(pairs, config.THRESHOLDS)
    best = metrics.select_threshold(sweep)
    ranking = metrics.ranking_metrics(pairs)
    history = fit["history"]
    result = {
        "study": study, "name": name, **hp,
        "epochs": fit["summary"]["epochs_completed"], "best_epoch": fit["summary"]["best_epoch"],
        "val_loss": fit["summary"]["best_validation_loss"],
        "final_train_loss": history["loss"][-1],
        "train_val_gap": history["val_loss"][fit["summary"]["best_epoch"] - 1] - history["loss"][fit["summary"]["best_epoch"] - 1],
        "threshold": best["threshold"], "val_precision": best["precision"], "val_recall": best["recall"],
        "val_f1": best["f1"], "val_iou": best["iou"], "val_auroc": ranking["auroc"], "val_ap": ranking["average_precision"],
        "seconds": time.perf_counter() - start,
    }
    training.write_json(result_path, result)
    # The checkpoint is only needed to score the run; drop it to keep the repository small.
    (run_dir / "best_model.keras").unlink(missing_ok=True)
    return result


def _records():
    from . import data, drive_data

    location = drive_data.ensure_dataset(allow_download=True)
    split = data.grouped_split(data.build_records(location.data_dir))
    return split["records"]["train"], split["records"]["validation"]


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
def run_study(study: dict, folder: str, train_records, val_records, force=False) -> list[dict]:
    rows = []
    for name, overrides in study.items():
        print(f"[{folder}] {name} ...", flush=True)
        row = run_variant(folder, name, overrides, train_records, val_records, force)
        print(f"    val F1 {row['val_f1']:.3f}  AUROC {row['val_auroc']:.3f}  epochs {row['epochs']}  {row['seconds']:.0f}s", flush=True)
        rows.append(row)
    return rows


def sample_search_space(n: int, seed: int = 0) -> dict:
    """Random search: learning rate log-uniform in [1e-4, 1e-2], batch size in {2, 4, 8}."""
    rng = np.random.default_rng(seed)
    return {
        f"coarse_{i + 1}": {
            "learning_rate": float(10 ** rng.uniform(-4, -2)),
            "batch_size": int(rng.choice([2, 4, 8])),
        }
        for i in range(n)
    }


def refine_search_space(best: dict, n: int = 3) -> dict:
    """Fine stage: log-uniform learning rates within a factor 2 of the best coarse trial, same batch size."""
    rng = np.random.default_rng(1)
    centre = math.log10(best["learning_rate"])
    return {
        f"fine_{i + 1}": {
            "learning_rate": float(10 ** (centre + rng.uniform(-math.log10(2), math.log10(2)))),
            "batch_size": int(best["batch_size"]),
        }
        for i in range(n)
    }


def run_search(train_records, val_records, n_coarse=5, n_fine=3, force=False) -> list[dict]:
    coarse = run_study(sample_search_space(n_coarse), "search", train_records, val_records, force)
    best = max(coarse, key=lambda r: r["val_f1"])
    print(f"[search] best coarse trial: {best['name']} (lr={best['learning_rate']:.2e}, batch={best['batch_size']})")
    fine = run_study(refine_search_space(best, n_fine), "search", train_records, val_records, force)
    return coarse + fine


def load_results(folder: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((EXPERIMENTS_DIR / folder).glob("*/result.json"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=["optimizers", "regularization", "search", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="retrain even when a cached result exists")
    args = parser.parse_args()

    train_records, val_records = _records()
    if args.study in ("optimizers", "all"):
        run_study(OPTIMIZER_STUDY, "optimizers", train_records, val_records, args.force)
    if args.study in ("regularization", "all"):
        run_study(REGULARIZATION_STUDY, "regularization", train_records, val_records, args.force)
    if args.study in ("search", "all"):
        run_search(train_records, val_records, force=args.force)


if __name__ == "__main__":
    main()
