"""Automatic road-alert logic on top of the segmentation models.

Pipeline (everything is automatic, the user only provides an image):

1. a segmentation model gives an anomaly score for every pixel;
2. the threshold fixed on the *validation* set turns the scores into a binary mask;
3. tiny speckles are removed and the remaining pixels are grouped into objects
   (connected components, nearby fragments are merged);
4. each object is cropped and classified by an **ImageNet-pretrained MobileNetV2**
   (the same network family as the transfer-learning encoder). ImageNet classes
   0-397 are animals, so the share of probability mass falling on them tells
   whether the object is an animal;
5. an alert level is produced: ``animal`` (alarm), ``obstacle`` (warning) or ``clear``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ImageNet class indices 0..397 are animals (fish, birds, reptiles, mammals, insects ...).
ANIMAL_CLASS_MAX = 397
ANIMAL_MASS_THRESHOLD = 0.50  # share of probability on animal classes: >= this -> animal
POSSIBLE_ANIMAL_THRESHOLD = 0.30  # between the two values -> "possible animal"
MIN_AREA_FRACTION = 0.0015  # objects smaller than 0.15 % of the image are treated as noise
SIGNIFICANT_AREA_FRACTION = 0.005  # an object must cover 0.5 % of the image ...
SIGNIFICANT_RELATIVE_AREA = 0.20  # ... and 20 % of the largest object to trigger an animal alert
MERGE_RADIUS = 2  # cells (on the low-resolution grid) used to join fragments of the same object
GRID_WIDTH = 320  # connected components are computed on a mask of about this width
CROP_MARGIN = 0.15
CLASSIFIER_SIZE = 224


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # y0, x0, y1, x1 (native resolution, x1/y1 exclusive)
    area_fraction: float
    mean_score: float
    label: str = "unknown"
    animal_probability: float = 0.0
    is_animal: bool = False
    possible_animal: bool = False
    significant: bool = True
    mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class AlertReport:
    level: str  # "animal" | "possible_animal" | "obstacle" | "clear"
    detections: list[Detection]
    mask: np.ndarray  # cleaned binary mask (native resolution)
    threshold: float

    @property
    def message(self) -> str:
        if self.level == "animal":
            names = ", ".join(sorted({d.label for d in self.detections if d.is_animal}))
            return f"ALERT: animal on the road ({names})"
        if self.level == "possible_animal":
            return "ALERT: possible animal on the road"
        if self.level == "obstacle":
            return "WARNING: unknown obstacle on the road"
        return "Road clear: nothing unexpected detected"


# ---------------------------------------------------------------------------
# Mask clean-up and connected components (pure numpy, no scipy/opencv needed)
# ---------------------------------------------------------------------------
def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labelling by breadth-first search (small grids only)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    height, width = mask.shape
    count = 0
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        count += 1
        labels[y, x] = count
        queue = deque([(y, x)])
        while queue:
            cy, cx = queue.popleft()
            for ny in range(max(cy - 1, 0), min(cy + 2, height)):
                for nx in range(max(cx - 1, 0), min(cx + 2, width)):
                    if mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = count
                        queue.append((ny, nx))
    return labels, count


def find_objects(mask: np.ndarray, min_area_fraction: float = MIN_AREA_FRACTION) -> list[np.ndarray]:
    """Split a binary mask into objects; returns one full-resolution boolean mask per object."""
    mask = mask.astype(bool)
    if not mask.any():
        return []
    height, width = mask.shape
    step = max(1, width // GRID_WIDTH)
    gh, gw = -(-height // step), -(-width // step)
    padded = np.zeros((gh * step, gw * step), dtype=bool)
    padded[:height, :width] = mask
    coarse = padded.reshape(gh, step, gw, step).any(axis=(1, 3))
    labels, count = _label(_dilate(coarse, MERGE_RADIUS))
    objects = []
    for index in range(1, count + 1):
        full = np.kron(labels == index, np.ones((step, step), dtype=bool))[:height, :width] & mask
        if full.mean() >= min_area_fraction:
            objects.append(full)
    objects.sort(key=lambda m: -m.sum())
    return objects


# ---------------------------------------------------------------------------
# Pretrained classifier (animal or not)
# ---------------------------------------------------------------------------
class AnimalClassifier:
    """ImageNet MobileNetV2 used on the crop of every detected object."""

    def __init__(self):
        import keras

        self._keras = keras
        self.model = keras.applications.MobileNetV2(weights="imagenet", input_shape=(CLASSIFIER_SIZE, CLASSIFIER_SIZE, 3))
        from keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

        self._decode, self._preprocess = decode_predictions, preprocess_input

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """Returns (best class name, probability mass on animal classes)."""
        import tensorflow as tf

        resized = tf.image.resize(crop.astype(np.float32), (CLASSIFIER_SIZE, CLASSIFIER_SIZE), method="bilinear")
        probs = np.asarray(self.model(self._preprocess(np.array(resized))[np.newaxis], training=False))[0]
        animal_mass = float(probs[: ANIMAL_CLASS_MAX + 1].sum())
        # name of the most probable class (animal class when the object is judged an animal)
        scope = probs[: ANIMAL_CLASS_MAX + 1] if animal_mass >= POSSIBLE_ANIMAL_THRESHOLD else probs
        one_hot = np.zeros_like(probs)
        one_hot[int(np.argmax(scope))] = 1.0
        name = self._decode(one_hot[np.newaxis], top=1)[0][0][1]
        return name.replace("_", " "), animal_mass


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def analyse(
    rgb: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    classifier: AnimalClassifier | None = None,
    min_area_fraction: float = MIN_AREA_FRACTION,
) -> AlertReport:
    height, width = scores.shape
    objects = find_objects(scores > threshold, min_area_fraction)
    detections = []
    for obj in objects:
        ys, xs = np.nonzero(obj)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        detection = Detection(
            bbox=(int(y0), int(x0), int(y1), int(x1)),
            area_fraction=float(obj.mean()),
            mean_score=float(scores[obj].mean()),
            mask=obj,
        )
        if classifier is not None:
            my, mx = int((y1 - y0) * CROP_MARGIN), int((x1 - x0) * CROP_MARGIN)
            crop = rgb[max(y0 - my, 0): min(y1 + my, height), max(x0 - mx, 0): min(x1 + mx, width)]
            detection.label, detection.animal_probability = classifier.classify(crop)
            detection.significant = (
                detection.area_fraction >= SIGNIFICANT_AREA_FRACTION
                and detection.area_fraction >= SIGNIFICANT_RELATIVE_AREA * objects[0].mean()
            )
            detection.is_animal = detection.significant and detection.animal_probability >= ANIMAL_MASS_THRESHOLD
            detection.possible_animal = (
                detection.significant and not detection.is_animal
                and detection.animal_probability >= POSSIBLE_ANIMAL_THRESHOLD
            )
            if not (detection.is_animal or detection.possible_animal):
                detection.label = "unknown object"
        detections.append(detection)

    mask = np.zeros(scores.shape, dtype=bool)
    for detection in detections:
        mask |= detection.mask
    level = "clear"
    if detections:
        if any(d.is_animal for d in detections):
            level = "animal"
        elif any(d.possible_animal for d in detections):
            level = "possible_animal"
        else:
            level = "obstacle"
    return AlertReport(level, detections, mask, float(threshold))


def draw_detections(rgb: np.ndarray, report: AlertReport, alpha: float = 0.45) -> np.ndarray:
    """Overlay of the cleaned mask with a bounding box per object (red = animal, orange = other)."""
    from PIL import Image, ImageDraw

    from .inference import overlay_mask

    out = overlay_mask(rgb, report.mask, alpha=alpha)
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    line = max(2, rgb.shape[1] // 300)
    for detection in report.detections:
        y0, x0, y1, x1 = detection.bbox
        color = (255, 0, 0) if detection.is_animal else (255, 165, 0)
        if not detection.significant:
            color = (200, 200, 200)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=line)
        text = detection.label if detection.label != "unknown" else "object"
        draw.text((x0 + line + 2, max(y0 - 12, 0) + line), text, fill=color)
    return np.asarray(image)
