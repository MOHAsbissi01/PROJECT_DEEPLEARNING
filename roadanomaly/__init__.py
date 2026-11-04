"""RoadAnomaly: road anomaly segmentation (Deep Learning module project).

Sub-modules
-----------
config      paths, environment detection and experimental protocol
drive_data  access to the dataset stored on Google Drive (Colab or local)
data        inventory, integrity checks, grouped split, preprocessing, tf.data
models      baseline CNN, MobileNetV2 U-Net (transfer), convolutional autoencoder
training    compilation, training with checkpoints, run bookkeeping
metrics     precision / recall / F1 / Dice / IoU, AUROC / AP / FPR@95
inference   experiment loading, prediction and visualisation helpers
"""

__version__ = "1.0.0"
