"""Dataset inventory, integrity checks, grouped splitting and preprocessing."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def mask_path_for(image_path: Path) -> Path:
    return image_path.with_suffix(".labels") / config.SEMANTIC_MASK_NAME


def load_frame_names(data_dir: Path) -> list[str]:
    names = json.loads((Path(data_dir) / "frame_list.json").read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("frame_list.json must be a non-empty list of file names.")
    if len(names) != len(set(names)):
        raise ValueError("Duplicated names in frame_list.json.")
    return names


def make_record(data_dir: Path, name: str) -> dict:
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"Unexpected file name: {name!r}")
    image_path = Path(data_dir) / "frames" / name
    return {"name": name, "image_path": image_path, "mask_path": mask_path_for(image_path)}


def build_records(data_dir: Path, names: list[str] | None = None) -> list[dict]:
    names = load_frame_names(data_dir) if names is None else names
    records = [make_record(data_dir, name) for name in names]
    missing = [str(r[k]) for r in records for k in ("image_path", "mask_path") if not r[k].is_file()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    return records


def filename_family(name: str) -> str:
    for prefix in config.FILENAME_FAMILIES:
        if name.startswith(prefix):
            return prefix
    return "other"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def load_native_mask(path: Path) -> np.ndarray:
    """Return the boolean anomaly mask at native resolution."""
    with Image.open(path) as mask:
        labels = np.array(mask)
    if labels.ndim != 2:
        raise ValueError(f"Mask must be 2-D: {path}")
    if not set(np.unique(labels).tolist()) <= config.ALLOWED_LABELS:
        raise ValueError(f"Undocumented labels in {path}: {np.unique(labels).tolist()}")
    return labels == config.ANOMALY_LABEL


def prepare_image(rgb: np.ndarray | Image.Image, height: int = config.IMG_HEIGHT, width: int = config.IMG_WIDTH) -> np.ndarray:
    """Resize (bilinear) and scale an RGB image to float32 in [0, 1]."""
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    image = image.convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_image_mask(record: dict, height: int = config.IMG_HEIGHT, width: int = config.IMG_WIDTH):
    """Image in [0, 1] (H, W, 3) and binary mask (H, W, 1); mask resized with nearest neighbour."""
    with Image.open(record["image_path"]) as image:
        image_array = prepare_image(image, height, width)
    native_mask = load_native_mask(record["mask_path"])
    resized = Image.fromarray(native_mask.astype(np.uint8)).resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    mask_array = np.asarray(resized, dtype=np.float32)[..., np.newaxis]
    return image_array, mask_array


# ---------------------------------------------------------------------------
# Exploration and integrity
# ---------------------------------------------------------------------------
def check_integrity(records: list[dict]) -> dict:
    """Validate every image/mask pair and collect dataset statistics."""
    errors, image_shapes, mask_shapes, label_pixels = [], Counter(), Counter(), Counter()
    per_image = []
    for record in records:
        try:
            with Image.open(record["image_path"]) as image:
                image.load()
                mode = image.mode
                image_array = np.array(image)
            with Image.open(record["mask_path"]) as mask:
                labels = np.array(mask)
            if mode != "RGB" or image_array.ndim != 3 or image_array.shape[-1] != 3:
                raise ValueError("image is not a 3-channel RGB image")
            if labels.ndim != 2:
                raise ValueError("mask is not 2-D")
            if image_array.shape[:2] != labels.shape:
                raise ValueError("image and mask sizes differ")
            values, counts = np.unique(labels, return_counts=True)
            if not set(values.tolist()) <= config.ALLOWED_LABELS:
                raise ValueError(f"undocumented labels {values.tolist()}")
            image_shapes[image_array.shape] += 1
            mask_shapes[labels.shape] += 1
            for value, count in zip(values, counts):
                label_pixels[int(value)] += int(count)
            anomaly_pixels = int(np.count_nonzero(labels == config.ANOMALY_LABEL))
            per_image.append(
                {
                    **record,
                    "family": filename_family(record["name"]),
                    "height": labels.shape[0],
                    "width": labels.shape[1],
                    "anomaly_pixels": anomaly_pixels,
                    "total_pixels": int(labels.size),
                    "anomaly_fraction": anomaly_pixels / labels.size,
                }
            )
        except (OSError, ValueError) as error:
            errors.append({"name": record["name"], "error": str(error)})

    total = sum(label_pixels.values())
    return {
        "valid_pairs": len(per_image),
        "errors": errors,
        "image_shapes": dict(image_shapes),
        "mask_shapes": dict(mask_shapes),
        "label_pixels": dict(label_pixels),
        "global_anomaly_fraction": label_pixels[config.ANOMALY_LABEL] / total if total else 0.0,
        "images_without_anomaly": sum(r["anomaly_pixels"] == 0 for r in per_image),
        "records": per_image,
    }


# ---------------------------------------------------------------------------
# Grouped split (prevents near-duplicate leakage)
# ---------------------------------------------------------------------------
def grouped_split(
    records: list[dict],
    scene_groups: dict[str, list[str]] = config.SCENE_GROUPS,
    seed: int = config.SEED,
    train_fraction: float = config.TRAIN_GROUP_FRACTION,
    val_fraction: float = config.VAL_GROUP_FRACTION,
) -> dict:
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("Invalid split fractions.")
    by_name = {r["name"]: r for r in records}
    if len(by_name) != len(records):
        raise ValueError("Duplicated names in the inventory.")

    group_by_name: dict[str, str] = {}
    for group_id, names in scene_groups.items():
        for name in names:
            if name not in by_name:
                raise ValueError(f"Grouped image absent from the dataset: {name}")
            if name in group_by_name:
                raise ValueError(f"Image declared in several groups: {name}")
            group_by_name[name] = group_id
    for name in sorted(by_name):
        group_by_name.setdefault(name, f"single::{name}")

    members = defaultdict(list)
    for name, group_id in group_by_name.items():
        members[group_id].append(name)

    group_ids = sorted(members)
    random.Random(seed).shuffle(group_ids)
    n_train = int(train_fraction * len(group_ids))
    n_val = int(val_fraction * len(group_ids))
    split_groups = {
        "train": group_ids[:n_train],
        "validation": group_ids[n_train : n_train + n_val],
        "test": group_ids[n_train + n_val :],
    }
    split_names = {s: sorted(n for g in groups for n in members[g]) for s, groups in split_groups.items()}
    validate_split(split_names, group_by_name, set(by_name))
    return {
        "group_by_name": group_by_name,
        "group_members": dict(members),
        "split_groups": split_groups,
        "split_names": split_names,
        "records": {s: [by_name[n] for n in names] for s, names in split_names.items()},
    }


def validate_split(split_names: dict, group_by_name: dict, all_names: set) -> None:
    sets = {s: set(split_names[s]) for s in SPLITS}
    if any(not s for s in sets.values()):
        raise ValueError("A partition is empty.")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Images shared by {a} and {b}: {sets[a] & sets[b]}")
        if {group_by_name[n] for n in sets[a]} & {group_by_name[n] for n in sets[b]}:
            raise ValueError(f"Scene groups shared by {a} and {b}.")
    if set().union(*sets.values()) != all_names:
        raise ValueError("Partitions do not cover the dataset exactly.")


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------
def augment(image, mask, seed=None):
    """Geometric + photometric augmentation; the same flip is applied to the mask."""
    import tensorflow as tf

    stacked = tf.concat([image, mask], axis=-1)
    stacked = tf.image.random_flip_left_right(stacked, seed=seed)
    image, mask = stacked[..., :3], stacked[..., 3:]
    image = tf.image.random_brightness(image, max_delta=0.10, seed=seed)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15, seed=seed)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15, seed=seed)
    return tf.clip_by_value(image, 0.0, 1.0), mask


def make_dataset(
    records: list[dict],
    training: bool = False,
    augmentation: bool = False,
    target: str = "mask",
    height: int = config.IMG_HEIGHT,
    width: int = config.IMG_WIDTH,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
):
    """Build a batched dataset.

    target="mask"           -> (image, mask) for supervised segmentation.
    target="reconstruction" -> (image, concat(image, mask)) for the autoencoder:
                               the mask lets the loss ignore anomalous pixels.
    """
    import tensorflow as tf

    records = tuple(records)

    def generate():
        for record in records:
            yield load_image_mask(record, height, width)

    dataset = tf.data.Dataset.from_generator(
        generate,
        output_signature=(
            tf.TensorSpec((height, width, 3), tf.float32),
            tf.TensorSpec((height, width, 1), tf.float32),
        ),
    )
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(records))).cache()
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
        if augmentation:
            dataset = dataset.map(lambda x, y: augment(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    if target == "reconstruction":
        dataset = dataset.map(lambda x, y: (x, tf.concat([x, y], axis=-1)))
    elif target != "mask":
        raise ValueError(f"Unknown target: {target}")
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
