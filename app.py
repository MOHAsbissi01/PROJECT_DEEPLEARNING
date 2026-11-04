"""Streamlit testing interface for the RoadAnomaly segmentation models.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("ROADANOMALY_PROJECT_ROOT", str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roadanomaly import config, data, drive_data, inference, metrics  # noqa: E402

MAX_UPLOAD_SIDE = 1920
METRIC_KEYS = ("precision", "recall", "f1", "dice", "iou")

st.set_page_config(page_title="RoadAnomaly Tester", page_icon="🚧", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def list_experiments() -> list[str]:
    paths = inference.discover_experiments(config.PROJECT_ROOT)
    output_dir = config.OUTPUT_DIR.resolve()
    if output_dir.is_dir() and not output_dir.is_relative_to(config.PROJECT_ROOT):
        paths += inference.discover_experiments(output_dir)
    return [str(p) for p in paths]


@st.cache_resource(show_spinner=False)
def get_experiment(path: str) -> inference.Experiment:
    return inference.load_experiment(Path(path))


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str):
    return inference.load_model(Path(path))


@st.cache_resource(show_spinner=False)
def get_dataset_location():
    try:
        return drive_data.find_local_dataset()
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def load_rgb(path: str) -> np.ndarray:
    return data.load_rgb(Path(path))


@st.cache_data(show_spinner=False)
def load_truth(path: str) -> np.ndarray:
    return data.load_native_mask(Path(path))


@st.cache_data(show_spinner="Running the model…", max_entries=64)
def predict(model_path: str, image_key: str, _rgb: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return inference.predict_probabilities(get_model(model_path), _rgb, image_size)


@st.cache_data(show_spinner=False, max_entries=16)
def reconstruct(model_path: str, image_key: str, _rgb: np.ndarray, image_size: tuple[int, int]):
    return inference.autoencoder_reconstruction(get_model(model_path), _rgb, image_size)


@st.cache_data(show_spinner="Computing dataset statistics…")
def dataset_statistics(data_dir: str) -> pd.DataFrame:
    report = data.check_integrity(data.build_records(Path(data_dir)))
    frame = pd.DataFrame(report["records"])
    frame["image_path"] = frame["image_path"].astype(str)
    frame["mask_path"] = frame["mask_path"].astype(str)
    return frame


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def split_of(name: str, splits: dict | None) -> str:
    if splits:
        for split, names in splits.items():
            if name in names:
                return split
    return "unknown"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚧 RoadAnomaly Tester")
experiment_paths = list_experiments()
if not experiment_paths:
    st.error(
        "No trained experiment found (`validation_selection.json` + `models/`). "
        "Run the notebook `RoadAnomaly_Project.ipynb` first, or copy a Colab experiment folder into the project."
    )
    st.stop()

experiment_path = st.sidebar.selectbox(
    "Experiment", experiment_paths, format_func=lambda p: get_experiment(p).label(config.PROJECT_ROOT)
)
experiment = get_experiment(experiment_path)
image_size = experiment.image_size

available_models = []
for name in experiment.model_names:
    try:
        experiment.checkpoint(name)
        available_models.append(name)
    except FileNotFoundError:
        st.sidebar.warning(f"Checkpoint missing for **{name}**")

selected_models = st.sidebar.multiselect(
    "Models", available_models, default=[experiment.primary_model] if experiment.primary_model in available_models else available_models[:1],
    key=f"models::{experiment.run_id}",
    help=f"Primary model selected on validation: {experiment.primary_model}",
)

with st.sidebar.expander("Decision thresholds", expanded=True):
    st.caption("Defaults = thresholds fixed on the validation set.")
    thresholds = {}
    for name in available_models:
        key = f"thr::{experiment.run_id}::{name}"
        if key not in st.session_state:
            st.session_state[key] = experiment.threshold(name)
        thresholds[name] = st.slider(name, 0.0, 1.0, step=0.01, key=key)
    if st.button("Reset to validation thresholds"):
        for name in available_models:
            st.session_state.pop(f"thr::{experiment.run_id}::{name}", None)
        st.rerun()

with st.sidebar.expander("Display"):
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)
    colormap = st.selectbox("Score colormap", ["inferno", "viridis", "magma", "jet", "turbo"])

location = get_dataset_location()
with st.sidebar.expander("Dataset (Google Drive)", expanded=location is None):
    if location:
        st.success("Dataset available")
        st.caption(f"{location.source}")
    else:
        st.warning("Dataset not found locally. You can still upload images.")
        st.caption(
            "Install Google Drive for Desktop, or add `.secrets/client_secret.json`, then download. "
            "See README → Dataset access."
        )
        if st.button("Download from Google Drive"):
            try:
                with st.spinner("Downloading from Google Drive (a browser window may open for sign-in)…"):
                    drive_data.ensure_dataset(allow_download=True, interactive=True)
                get_dataset_location.clear()
                st.rerun()
            except Exception as error:  # noqa: BLE001 - shown to the user
                st.error(str(error))

DATA_DIR = location.data_dir if location else None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_test, tab_results, tab_dataset, tab_about = st.tabs(
    ["🔍 Test an image", "📊 Experiment results", "🗂️ Dataset explorer", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# 1. Test an image
# ---------------------------------------------------------------------------
with tab_test:
    st.header("Detect anomalies in a road image")
    sources = ["Dataset image", "Upload an image"] if DATA_DIR else ["Upload an image"]
    source = st.radio("Image source", sources, horizontal=True)

    rgb, truth, image_key, image_label = None, None, None, None
    if source == "Dataset image":
        frame = dataset_statistics(str(DATA_DIR))
        frame["split"] = [split_of(n, experiment.splits) for n in frame["name"]]
        c1, c2, c3 = st.columns([1, 1, 3])
        split_choice = c1.selectbox("Split", ["test", "validation", "train", "all"],
                                    help="Only test images give an unbiased view of performance.")
        family_choice = c2.selectbox("Family", ["all", *sorted(frame["family"].unique())])
        subset = frame if split_choice == "all" else frame[frame["split"] == split_choice]
        if family_choice != "all":
            subset = subset[subset["family"] == family_choice]
        if subset.empty:
            st.info("No image matches these filters.")
        else:
            image_label = c3.selectbox("Image", subset["name"].tolist())
            row = subset[subset["name"] == image_label].iloc[0]
            rgb, truth, image_key = load_rgb(row["image_path"]), load_truth(row["mask_path"]), row["image_path"]
            if split_choice == "train" or row["split"] == "train":
                st.caption("⚠️ Training image: the models have seen it, so scores are optimistic.")
    elif source == "Upload an image":
        c1, c2 = st.columns(2)
        upload = c1.file_uploader("Road image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        mask_upload = c2.file_uploader(
            "Optional ground-truth mask (PNG: 0/2 labels or black/white)", type=["png"],
            help="If provided, precision / recall / F1 / IoU are computed for your image.",
        )
        if upload is not None:
            raw = upload.getvalue()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if max(image.size) > MAX_UPLOAD_SIDE:
                image.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.Resampling.LANCZOS)
            rgb, image_key, image_label = np.array(image), hashlib.sha1(raw).hexdigest(), upload.name
            if mask_upload is not None:
                labels = np.array(Image.open(mask_upload).convert("L"))
                values = set(np.unique(labels).tolist())
                truth = labels == config.ANOMALY_LABEL if values <= config.ALLOWED_LABELS else labels > 127
                if truth.shape != rgb.shape[:2]:
                    truth = np.array(
                        Image.fromarray(truth.astype(np.uint8)).resize(rgb.shape[1::-1], Image.Resampling.NEAREST)
                    ).astype(bool)

    if rgb is not None:
        if not selected_models:
            st.warning("Select at least one model in the sidebar.")
        top = st.columns(2)
        top[0].image(rgb, caption=f"Input: {image_label} ({rgb.shape[1]}×{rgb.shape[0]})", width="stretch")
        if truth is not None:
            top[1].image(inference.overlay_mask(rgb, truth, alpha=alpha),
                         caption=f"Ground truth: anomaly area {truth.mean():.2%}", width="stretch")
        else:
            top[1].info("No ground truth: only the predictions are shown.")

        comparison = []
        for name in selected_models:
            checkpoint = str(experiment.checkpoint(name))
            scores = predict(checkpoint, f"{image_key}", rgb, image_size)
            threshold = thresholds[name]
            predicted = scores > threshold
            st.subheader(f"{name}")
            st.caption(config.MODEL_DESCRIPTIONS.get(name, ""))

            cols = st.columns(3)
            cols[0].image(inference.blend(rgb, inference.colorize(scores, colormap), 0.6),
                          caption=f"Anomaly score (max {scores.max():.2f})", width="stretch")
            cols[1].image(inference.overlay_mask(rgb, predicted, alpha=alpha),
                          caption=f"Prediction at threshold {threshold:.2f}: {predicted.mean():.2%} of pixels",
                          width="stretch")
            if truth is not None:
                cols[2].image(inference.error_map(truth, predicted),
                              caption="🟩 TP  🟥 FP (false alarm)  🟦 FN (missed)", width="stretch")
            else:
                cols[2].image((predicted * 255).astype(np.uint8), caption="Binary mask", width="stretch")

            if name == "autoencoder":
                recon = reconstruct(checkpoint, f"{image_key}", rgb, image_size)
                if recon is not None:
                    with st.expander("Autoencoder reconstruction"):
                        st.image(recon, caption="What the autoencoder 'expects' the road to look like", width="stretch")

            row = {"model": name, "threshold": threshold, "predicted area": predicted.mean()}
            if truth is not None:
                scores_dict = metrics.image_scores(truth, scores, threshold)
                row.update({k: scores_dict[k] for k in METRIC_KEYS})
                mcols = st.columns(5)
                for col, key in zip(mcols, METRIC_KEYS):
                    col.metric(key.upper() if key in ("f1", "iou") else key.capitalize(), f"{scores_dict[key]:.3f}")
            comparison.append(row)

            st.download_button(
                f"Download {name} mask (PNG)", png_bytes((predicted * 255).astype(np.uint8)),
                file_name=f"{Path(image_label).stem}_{name}_mask.png", mime="image/png", key=f"dl::{name}",
            )
            st.divider()

        if len(comparison) > 1:
            st.subheader("Model comparison on this image")
            st.dataframe(pd.DataFrame(comparison).set_index("model").style.format("{:.3f}"))

# ---------------------------------------------------------------------------
# 2. Experiment results
# ---------------------------------------------------------------------------
with tab_results:
    st.header(f"Experiment {experiment.run_id}")
    st.caption(str(experiment.directory))
    c = st.columns(4)
    c[0].metric("Primary model", experiment.primary_model)
    c[1].metric("Input size", f"{image_size[1]}×{image_size[0]}")
    c[2].metric("Models", len(experiment.model_names))
    c[3].metric("Test evaluated", "yes" if experiment.test_results else "no")

    if experiment.splits:
        st.write("**Partitions:** " + ", ".join(f"{s}: {len(n)} images" for s, n in experiment.splits.items()))

    st.subheader("Validation (used to select thresholds)")
    val_rows = []
    for name, info in experiment.payload["models"].items():
        row = {"model": name, "threshold": info["threshold"], **{k: info["validation_metrics"][k] for k in METRIC_KEYS}}
        row.update(info.get("ranking_metrics", {}))
        val_rows.append(row)
    st.dataframe(pd.DataFrame(val_rows).set_index("model").style.format("{:.4f}"))

    curves = {}
    for name in experiment.model_names:
        csv_path = experiment.directory / f"{name}_thresholds.csv"
        if csv_path.is_file():
            curves[name] = pd.read_csv(csv_path)
    if curves:
        st.markdown("**Validation metrics vs. threshold**")
        cols = st.columns(len(curves))
        for col, (name, frame) in zip(cols, curves.items()):
            col.caption(name)
            col.line_chart(frame.set_index("threshold")[["precision", "recall", "f1", "iou"]], height=250)

    if experiment.test_results:
        st.subheader("Test set (final, unbiased evaluation)")
        test_rows = []
        for name, result in experiment.test_results["models"].items():
            row = {"model": name, "threshold": result["threshold"], **{k: result["global_metrics"][k] for k in METRIC_KEYS}}
            row.update(result.get("ranking_metrics", {}))
            test_rows.append(row)
        st.dataframe(pd.DataFrame(test_rows).set_index("model").style.format("{:.4f}"))

        per_image = pd.DataFrame({
            name: {r["name"]: r["f1"] for r in result["per_image_metrics"]}
            for name, result in experiment.test_results["models"].items()
        })
        st.markdown("**Per-image F1 on test**")
        st.bar_chart(per_image, horizontal=True, stack=False, height=420)

        family_rows = [
            {"model": name, "family": fam, **{k: s[k] for k in ("images", "f1", "iou")}}
            for name, result in experiment.test_results["models"].items()
            for fam, s in (result.get("family_metrics") or {}).items()
        ]
        if family_rows:
            st.markdown("**Per family**")
            st.dataframe(pd.DataFrame(family_rows).pivot(index="family", columns="model", values=["f1", "iou"]).style.format("{:.3f}"))

    figures = sorted(experiment.directory.glob("*.png"))
    eval_dir = experiment.directory / "final_evaluation"
    figures += sorted(eval_dir.glob("*.png")) + sorted((eval_dir / "analysis").glob("*.png"))
    if figures:
        st.subheader("Saved figures")
        for figure in figures:
            with st.expander(figure.stem.replace("_", " ")):
                st.image(str(figure), width="stretch")

# ---------------------------------------------------------------------------
# 3. Dataset explorer
# ---------------------------------------------------------------------------
with tab_dataset:
    st.header("RoadAnomaly dataset")
    if not DATA_DIR:
        st.info("The dataset is not available locally: see the sidebar to fetch it from Google Drive.")
    else:
        frame = dataset_statistics(str(DATA_DIR))
        frame["split"] = [split_of(n, experiment.splits) for n in frame["name"]]
        total = frame["total_pixels"].sum()
        c = st.columns(4)
        c[0].metric("Images", len(frame))
        c[1].metric("Anomaly pixels", f"{frame['anomaly_pixels'].sum() / total:.2%}")
        c[2].metric("Median anomaly area", f"{frame['anomaly_fraction'].median():.2%}")
        c[3].metric("Native size", f"{frame['width'].iloc[0]}×{frame['height'].iloc[0]}")

        left, right = st.columns(2)
        left.markdown("**Images per family and split**")
        left.bar_chart(frame.groupby(["family", "split"]).size().unstack(fill_value=0))
        right.markdown("**Anomaly area per image (%)**")
        bins = pd.cut(frame["anomaly_fraction"] * 100, bins=[0, 1, 2, 5, 10, 20, 30, 50])
        right.bar_chart(bins.value_counts().sort_index().rename(lambda b: str(b)))

        st.dataframe(
            frame[["name", "family", "split", "anomaly_fraction"]].sort_values("name"),
            hide_index=True,
            column_config={"anomaly_fraction": st.column_config.ProgressColumn("anomaly area", format="percent", min_value=0, max_value=0.5)},
            height=300,
        )
        chosen = st.selectbox("Inspect an image", frame["name"].sort_values().tolist())
        row = frame[frame["name"] == chosen].iloc[0]
        rgb = load_rgb(row["image_path"])
        truth = load_truth(row["mask_path"])
        cols = st.columns(3)
        cols[0].image(rgb, caption=chosen, width="stretch")
        cols[1].image((truth * 255).astype(np.uint8), caption="Ground-truth mask", width="stretch")
        cols[2].image(inference.overlay_mask(rgb, truth, alpha=alpha), caption=f"Overlay ({row['split']})", width="stretch")

# ---------------------------------------------------------------------------
# 4. About
# ---------------------------------------------------------------------------
with tab_about:
    st.header("About")
    st.markdown(
        """
**Task.** Binary semantic segmentation of unexpected road obstacles (animals, rocks, cones, lost cargo…)
on the *RoadAnomaly* dataset (Lis et al., *Detecting the Unexpected via Image Resynthesis*, ICCV 2019).
Every model outputs an anomaly score in [0, 1] for each pixel; a pixel is flagged when its score is **above the threshold**.

**Models**
"""
        + "\n".join(f"* **{name}**: {text}" for name, text in config.MODEL_DESCRIPTIONS.items())
        + """

**Metrics** (computed at native resolution against the original mask)
* *Precision* = TP / (TP + FP): share of flagged pixels that are really anomalous (few false alarms).
* *Recall* = TP / (TP + FN): share of anomalous pixels that are detected (few misses).
* *F1 = Dice* = 2TP / (2TP + FP + FN); *IoU* = TP / (TP + FP + FN).
* *AUROC / AP / FPR@95TPR*: threshold-free ranking quality of the scores (as in the reference paper).

**Data.** The dataset stays on Google Drive and is read through Google Drive for Desktop or
downloaded once into `data/` with the Drive API (see `README.md`).
"""
    )
