"""Central configuration shared by the notebook, the training code and the Streamlit app.

Every value that can differ between machines (data location, output folder,
Google Drive folder ID) can be overridden with an environment variable, so the
same code runs unchanged in VS Code, in Google Colab and inside Streamlit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Execution environment
# ---------------------------------------------------------------------------
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ

# Root of the project (the folder that contains the `roadanomaly` package).
PROJECT_ROOT = Path(
    os.environ.get("ROADANOMALY_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).resolve()

# ---------------------------------------------------------------------------
# Data source: Google Drive
# ---------------------------------------------------------------------------
# Shared Drive folders (anyone with the link), tried in order until one holds the COMPLETE dataset
# (60 images + 60 semantic masks).  Each entry: folder id + path of sub-folders down to the one
# that contains RoadAnomaly/{frames, frame_list.json, ...}.
_env_id = os.environ.get("ROADANOMALY_DRIVE_FOLDER_ID")
if _env_id:
    DRIVE_SOURCES = [{
        "id": _env_id,
        "subpath": tuple(p for p in os.environ.get("ROADANOMALY_DRIVE_SUBPATH", "RoadAnomaly").split(",") if p),
    }]
else:
    DRIVE_SOURCES = [
        {"id": "1P_PpgTB_yWJ3zvzpYnfXSQZ83jgPiv44", "subpath": ("RoadAnomaly",)},  # professor's reference version
        {"id": "1SWjnBFraWwCA6RSy_JIWKqPCA9uPuUQL", "subpath": ("RoadAnomaly",)},  # original Colab folder
    ]
DRIVE_FOLDER_ID = DRIVE_SOURCES[0]["id"]
DRIVE_FOLDER_NAME = "project-deeplearning"

# Local cache where the Drive dataset is downloaded when it is not reachable
# through Google Drive for Desktop.  In Colab the fast local disk is used.
DATA_CACHE_DIR = Path(
    os.environ.get(
        "ROADANOMALY_CACHE_DIR",
        "/content/roadanomaly_data" if IN_COLAB else PROJECT_ROOT / "data",
    )
)

# Optional explicit dataset folder (the one containing frames/ and frame_list.json).
DATA_DIR_OVERRIDE = os.environ.get("ROADANOMALY_DATA_DIR")

# Credentials for the local Google Drive API fallback (never commit this folder).
SECRETS_DIR = Path(os.environ.get("ROADANOMALY_SECRETS_DIR", PROJECT_ROOT / ".secrets"))
OAUTH_CLIENT_SECRET = SECRETS_DIR / "client_secret.json"
OAUTH_TOKEN = SECRETS_DIR / "token.json"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # only for publicly shared folders

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("ROADANOMALY_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# ---------------------------------------------------------------------------
# Dataset conventions (see RoadAnomaly/readme.md)
# ---------------------------------------------------------------------------
BACKGROUND_LABEL = 0
ANOMALY_LABEL = 2
ALLOWED_LABELS = {BACKGROUND_LABEL, ANOMALY_LABEL}
SEMANTIC_MASK_NAME = "labels_semantic.png"
FILENAME_FAMILIES = ("animals", "cones", "lost_tires", "obstacles", "vehicle")

# ---------------------------------------------------------------------------
# Experimental protocol (identical to the original Colab experiment)
# ---------------------------------------------------------------------------
SEED = 42
IMG_HEIGHT = 360  # native frames are 720 x 1280 -> factor 2 downsampling
IMG_WIDTH = 640
BATCH_SIZE = 4

TRAIN_GROUP_FRACTION = 0.70
VAL_GROUP_FRACTION = 0.15

# Images showing the same scene / series must stay in the same partition,
# otherwise near-duplicates leak from train into validation/test.
SCENE_GROUPS: dict[str, list[str]] = {
    "lost_tires": [
        "lost_tires03.webp",
        "lost_tires08.webp",
        "lost_tires09.webp",
    ],
    "cardboard_obstacles": [
        "obstacles03_cardboard_czech_hedgehog.webp",
        "obstacles04_cardboard_czech_hedgehog.webp",
        "obstacles05_cardboard_czech_hedgehog.webp",
        "obstacles06_cardboard_czech_hedgehog.webp",
        "obstacles07_czech_hedgehog.webp",
    ],
    "rocks_series": [
        "obstacles12_rocks2.webp",
        "obstacles12_rocks4.webp",
        "obstacles12_rocks5.webp",
        "obstacles12_rocks6.webp",
    ],
    "broadmoor": [
        "cones02_Broadmoor_Manhole_Cones.webp",
        "vehicle02_Broadmoor_Cones_Skidloader.webp",
    ],
}

# Decision thresholds examined on the validation set.
THRESHOLDS = np.round(np.linspace(0.05, 0.95, 19), 2)

# Human-readable model descriptions (used by the report and the Streamlit UI).
MODEL_DESCRIPTIONS = {
    "baseline": "Small encoder-decoder CNN trained from scratch (supervised segmentation).",
    "transfer": "U-Net with an ImageNet-pretrained MobileNetV2 encoder (transfer learning).",
    "autoencoder": (
        "Convolutional autoencoder trained to reconstruct road background only; "
        "the per-pixel reconstruction error is the anomaly score (resynthesis idea, Lis et al. 2019)."
    ),
}
