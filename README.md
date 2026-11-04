# Road Anomaly Detection (RoadAnomaly dataset)

Deep Learning project: pixel-level detection of unexpected obstacles on the road
(animals, rocks, cones, lost cargo) with three complementary approaches:

| Model | Idea |
|---|---|
| `baseline` | small encoder–decoder CNN trained from scratch |
| `transfer` | U-Net on an ImageNet-pretrained **MobileNetV2** encoder (transfer learning) |
| `autoencoder` | convolutional autoencoder trained on background pixels; reconstruction error = anomaly score (resynthesis idea of Lis et al., ICCV 2019) |

The dataset stays on **Google Drive**; the code runs in **VS Code**, in **Colab**, and is exposed through a **Streamlit** testing interface.

## Layout

```
RoadAnomaly_Project.ipynb   main notebook (exploration → report), VS Code / Colab
app.py                      Streamlit testing interface
roadanomaly/                reusable package
  config.py                   paths, protocol, scene groups (env-var overridable)
  drive_data.py               Google Drive access (Drive for Desktop / Drive API / Colab)
  data.py                     inventory, integrity checks, grouped split, tf.data
  models.py                   CNN, MobileNetV2 U-Net, autoencoder (+ losses)
  training.py                 compile / fit / checkpoints / history
  metrics.py                  precision, recall, F1, Dice, IoU, AUROC, AP, FPR@95
  inference.py                experiment loading, prediction, visualisation
  report.py                   technical-report generator
DEEPLEARING_FINALE.ipynb    original Colab notebook (kept for reference)
2026…/ technical_report/    outputs of the original Colab run (reused automatically)
```

## Quick start

```powershell
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py          # test the already-trained models
```

For the notebook, open `RoadAnomaly_Project.ipynb` in VS Code, pick the `.venv` kernel and *Run All*.
On native Windows TensorFlow runs on CPU (≥ 2.11 has no Windows GPU support); use WSL2 or Colab for GPU training.

## Dataset access (Google Drive)

Sources are listed in `config.DRIVE_SOURCES` and tried in order until one holds the **complete** dataset
(60 images + 60 semantic masks; verified against `frame_list.json`). An incomplete copy is never used silently.

| Priority | Drive folder | Status |
|---|---|---|
| 1 | `1P_PpgTB_yWJ3zvzpYnfXSQZ83jgPiv44` (`RoadAnomaly/RoadAnomaly`), professor's reference version | complete (677 files, 60/60 masks) |
| 2 | `1SWjnBFraWwCA6RSy_JIWKqPCA9uPuUQL` (original Colab folder) | complete, used only as fallback |

Each source has its own cache folder (`data/drive_<id>/`), so versions never mix.

`drive_data.ensure_dataset()` looks for the dataset in this order:

1. **`ROADANOMALY_DATA_DIR`**: explicit path to the folder containing `frames/` and `frame_list.json`.
2. **Local cache** `data/` (or `/content/roadanomaly_data` in Colab), filled by an earlier download.
3. **Google Drive for Desktop**: *Add shortcut to Drive* on the shared folder; files are read from Drive directly. Use `ROADANOMALY_DRIVE_PATH` if your mount is elsewhere.
4. **Download into the cache** (incremental, only missing files):
   * folders shared as *anyone with the link*: **no credentials needed** (public listing + download);
   * *Colab*: automatic sign-in (`google.colab.auth`);
   * private folders locally: OAuth client of type **Desktop app** (Drive API enabled) saved as `.secrets/client_secret.json`, or `GOOGLE_API_KEY`.
5. Override the source with `ROADANOMALY_DRIVE_FOLDER_ID` (+ `ROADANOMALY_DRIVE_SUBPATH`, default `project-deeplearning,RoadAnomaly`).

`.secrets/` and `data/` are git-ignored; never share the credentials.

## Running in Colab

Put this project folder in your Drive (`MyDrive/PROJECT-DEEPLEARNING`, or edit `COLAB_CODE_DIR` in the first cell), open the notebook in Colab, and run it: Drive is mounted, the dataset is downloaded to the fast local disk, and a GPU speeds up training. Download `outputs/model_selection/<run>/` afterwards (or keep it on Drive) so Streamlit can use it.

## Notebook workflow

| Step | Assignment requirement |
|---|---|
| 1–2 Setup, Drive access | reproducibility |
| 3 Exploration (integrity, class imbalance, image/mask overlays, near-duplicates) | dataset exploration |
| 4–5 Grouped split (no near-duplicate leakage), preprocessing, augmentation | data preprocessing |
| 6–7 CNN, MobileNetV2 U-Net (+ optional fine-tuning), autoencoder; training | model implementation |
| 8–9 Threshold selection on validation; test: precision, recall, F1, Dice, IoU, AUROC, AP | performance evaluation |
| 10 Per-family / per-image analysis, error maps, discussion | results analysis |
| 11 Report generation → `technical_report/technical_report.md` | technical report |

`TRAINING_PLAN` (section 7) decides per model whether to **reuse** the Colab checkpoint (same partitions required, verified automatically) or **train**. By default `baseline` and `transfer` are reused and the autoencoder is trained. Environment switches: `ROADANOMALY_FORCE_TRAIN=1` retrains everything, `ROADANOMALY_QUICK_RUN=1` runs 1 epoch per model (smoke test).

## Streamlit interface

`streamlit run app.py`

* **Test an image**: pick a dataset image (filter by split/family) or upload your own (optionally with a ground-truth mask); choose models, move the threshold slider live; see score heatmap, predicted mask, error map (TP/FP/FN) and per-image metrics; download masks.
* **Experiment results**: validation/test tables, threshold curves, per-image and per-family scores, saved figures.
* **Dataset explorer**: statistics and image/mask inspection.

The app discovers every experiment folder (`validation_selection.json` + `models/`) under the project, including the original Colab output, and falls back to upload-only mode if the dataset is not reachable.

## Protocol notes

* Split by **scene groups** (seed 42, 70/15/15 %), identical to the Colab run; test used once with thresholds frozen on validation.
* Evaluation at native 1280×720 against the original masks; accuracy is not reported as a headline metric (a background-only model scores ≈ 90 %).
* RoadAnomaly was designed as a test benchmark; training on part of it is a pedagogical supervised protocol and is not comparable to the paper's open-set benchmark.

## Reference

K. Lis, K. Nakka, P. Fua, M. Salzmann. *Detecting the Unexpected via Image Resynthesis.* ICCV 2019.
