"""Road anomaly alert: upload a road image, everything else is automatic.

Run from the project root:
    streamlit run app.py

The advanced lab (experiments, thresholds, dataset explorer, all models) is app_advanced.py.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("ROADANOMALY_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadanomaly import alert, config, drive_data, inference  # noqa: E402

MAX_UPLOAD_SIDE = 1920
MODEL_CHOICES = {
    "Automatic (best model chosen on validation)": None,
    "Pre-trained model (MobileNetV2 U-Net, transfer learning)": "transfer",
    "Course model (CNN from scratch, chapter 4)": "baseline",
}

st.set_page_config(page_title="Road Alert", page_icon="🚧", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_experiment() -> inference.Experiment | None:
    paths = inference.discover_experiments(config.PROJECT_ROOT)
    return inference.load_experiment(paths[0]) if paths else None


@st.cache_resource(show_spinner="Loading the segmentation model…")
def get_model(path: str):
    return inference.load_model(Path(path))


@st.cache_resource(show_spinner="Loading the pre-trained animal classifier…")
def get_classifier() -> alert.AnimalClassifier:
    return alert.AnimalClassifier()


@st.cache_resource(show_spinner=False)
def get_examples() -> dict[str, str]:
    """Dataset images (if the dataset is available locally) offered as ready-made examples."""
    try:
        from roadanomaly import data

        location = drive_data.find_local_dataset()
        return {r["name"]: str(r["image_path"]) for r in data.build_records(location.data_dir)}
    except Exception:  # noqa: BLE001 - examples are optional
        return {}


@st.cache_data(show_spinner=False)
def beep() -> bytes:
    """Two-tone warning sound (generated, no audio file needed)."""
    rate = 22050
    t = np.arange(int(rate * 0.25)) / rate
    tone = np.concatenate([np.sin(2 * np.pi * f * t) for f in (880, 660, 880, 660)])
    samples = (tone * 0.5 * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner="Analysing the image…", max_entries=32)
def run_pipeline(model_path: str, threshold: float, image_size: tuple[int, int], image_key: str, _rgb: np.ndarray):
    scores = inference.predict_probabilities(get_model(model_path), _rgb, image_size)
    report = alert.analyse(_rgb, scores, threshold, get_classifier())
    return scores, report


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
experiment = get_experiment()
st.title("🚧 Road Alert")
st.caption("Upload a photo of a road. The system finds unexpected objects and raises an alert when an animal is on the road.")

if experiment is None:
    st.error(
        "No trained model found. Run the notebook `RoadAnomaly_Project.ipynb` first "
        "(or copy an experiment folder with `validation_selection.json` and `models/` into the project)."
    )
    st.stop()

with st.sidebar:
    st.header("Options")
    choice = st.radio("Segmentation model", list(MODEL_CHOICES), help="Automatic = the model with the best validation F1.")
    model_name = MODEL_CHOICES[choice] or experiment.primary_model
    if model_name not in experiment.model_names:
        st.warning(f"`{model_name}` is not in this experiment, using `{experiment.primary_model}`.")
        model_name = experiment.primary_model
    st.caption(config.MODEL_DESCRIPTIONS.get(model_name, ""))
    st.caption(f"Decision threshold {experiment.threshold(model_name):.2f} (fixed on the validation set; nothing to tune).")
    sound = st.toggle("Alarm sound", value=True)
    st.divider()
    st.caption("Advanced lab (metrics, thresholds, dataset): `streamlit run app_advanced.py`")

rgb, image_key, image_label = None, None, None
upload = st.file_uploader("Road image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if upload is not None:
    raw = upload.getvalue()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if max(image.size) > MAX_UPLOAD_SIDE:
        image.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.Resampling.LANCZOS)
    rgb, image_key, image_label = np.array(image), hashlib.sha1(raw).hexdigest(), upload.name
else:
    examples = get_examples()
    if examples:
        with st.expander("No image at hand? Try a dataset example"):
            name = st.selectbox("Example", ["—", *sorted(examples)])
            if name != "—":
                from roadanomaly import data

                rgb, image_key, image_label = data.load_rgb(Path(examples[name])), examples[name], name

if rgb is None:
    st.info("Waiting for an image…")
    st.stop()

try:
    checkpoint = str(experiment.checkpoint(model_name))
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

scores, report = run_pipeline(checkpoint, experiment.threshold(model_name), experiment.image_size, f"{image_key}", rgb)

# ---- verdict -------------------------------------------------------------------
if report.level == "animal":
    st.error(f"🚨 {report.message}", icon="🚨")
elif report.level == "possible_animal":
    st.error(f"🚨 {report.message}", icon="🚨")
elif report.level == "obstacle":
    st.warning(f"⚠️ {report.message}", icon="⚠️")
else:
    st.success(f"✅ {report.message}", icon="✅")
if sound and report.level in ("animal", "possible_animal"):
    st.audio(beep(), format="audio/wav", autoplay=True)

# ---- images ---------------------------------------------------------------------
left, right = st.columns(2)
left.image(rgb, caption=f"Input: {image_label} ({rgb.shape[1]}×{rgb.shape[0]})", width="stretch")
annotated = alert.draw_detections(rgb, report)
right.image(annotated, caption="Detected anomalies (red box = animal, orange = other object)", width="stretch")

if report.detections:
    st.subheader("Detected objects")
    rows = [
        {
            "object": d.label if (d.is_animal or d.possible_animal) else "unknown object",
            "type": "animal" if d.is_animal else "possible animal" if d.possible_animal else "obstacle",
            "animal probability": round(d.animal_probability, 2),
            "area (% of image)": round(100 * d.area_fraction, 2),
            "mean anomaly score": round(d.mean_score, 2),
            "counted": "yes" if d.significant else "no (too small)",
        }
        for d in report.detections
    ]
    st.dataframe(rows, hide_index=True, width="stretch")

with st.expander("How the decision was made"):
    c1, c2 = st.columns(2)
    c1.image(inference.blend(rgb, inference.colorize(scores, "inferno"), 0.6),
             caption=f"Pixel anomaly score from the {model_name} model (max {scores.max():.2f})", width="stretch")
    c2.image((report.mask * 255).astype(np.uint8), caption=f"Mask after threshold {report.threshold:.2f} and clean-up", width="stretch")
    st.markdown(
        f"""
1. **Segmentation** (`{model_name}`) gives an anomaly score for every pixel.
2. The **validation-selected threshold ({report.threshold:.2f})** turns it into a mask.
3. Specks smaller than {alert.MIN_AREA_FRACTION:.2%} of the image are removed and the remaining pixels are grouped into objects.
4. Each object is cropped and classified by an **ImageNet-pretrained MobileNetV2**; if at least
   {alert.ANIMAL_MASS_THRESHOLD:.0%} of its probability falls on animal classes it is an **animal**
   ({alert.POSSIBLE_ANIMAL_THRESHOLD:.0%}–{alert.ANIMAL_MASS_THRESHOLD:.0%}: *possible animal*).
   Only objects covering ≥ {alert.SIGNIFICANT_AREA_FRACTION:.1%} of the image and ≥ {alert.SIGNIFICANT_RELATIVE_AREA:.0%} of the largest object can trigger the animal alarm.
5. Alert level: animal / possible animal → 🚨, other object → ⚠️, nothing → ✅.

*Limits:* trained on 39 images only; the classifier knows ImageNet animals (not every species) and may take a
rock or a hedgehog barrier for an animal; the model does not know where the road is.
"""
    )

st.download_button("Download the annotated image (PNG)", png_bytes(annotated),
                   file_name=f"{Path(image_label).stem}_alert.png", mime="image/png")
