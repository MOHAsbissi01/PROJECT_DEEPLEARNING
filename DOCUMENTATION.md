# Road Anomaly Detection — Project Documentation

*Deep Learning module · RoadAnomaly dataset · pixel-level detection of unexpected obstacles on the road*

This document explains **everything** in the project so that any question can be answered: what was built, why, how it
maps to the course, the maths behind each choice, the results (with their limits) and a question-and-answer bank for the
oral. Section 12 is a quick cheat-sheet; sections 13–14 are the Q&A.

**Contents**
1. The project in one page
2. Repository map and how to run it
3. How the project follows the course (chapter by chapter)
4. Data: the RoadAnomaly dataset, exploration, split, preprocessing
5. The three models in detail (+ the maths of every layer)
6. The autoencoder explained from scratch (not in the course)
7. Losses, optimisers, regularisation, hyper-parameters (course chapter 3)
8. Evaluation protocol and metrics
9. Results
10. Critical analysis: what works, what does not, and why
11. Limitations and what we would do next
12. Cheat-sheet of numbers
13. Questions on the course concepts (with answers)
14. Questions on the project (with answers)
15. Honest list of weaknesses (say them before the professor finds them)

---

## 1. The project in one page

**Problem.** A self-driving car is trained on known classes (road, car, pedestrian…). In the real world it meets things it was never
taught: a cow, a rock, a lost tyre, a traffic cone, cardboard "hedgehog" obstacles. The task is to flag, **for every pixel**,
whether it belongs to such an **unexpected obstacle** (anomaly) or to the normal scene (background).

**Data.** *RoadAnomaly* (Lis et al., ICCV 2019): **60 photographs** (1280 × 720) of roads with an unexpected object, each with a
hand-made mask (0 = background, 2 = anomaly). Anomalies cover ≈ 9.85 % of the pixels. It was designed as a **test benchmark**,
not as a training set; we use part of it for training as a *pedagogical supervised protocol* (this is stated in every report).

**Approach.** Three models that share the same input/output contract (RGB image → anomaly-score map in [0, 1]) and are evaluated
with the same protocol:

| Model | Idea | Course link |
|---|---|---|
| **baseline** | small encoder–decoder CNN trained **from scratch** (46 689 parameters) | CNN (ch. 4), BCE (ch. 2) |
| **transfer** | **U-Net** whose encoder is **MobileNetV2 pre-trained on ImageNet** (3.33 M parameters, 2.69 M trainable after fine-tuning) | CNN architectures (ch. 4), regularisation by pre-training |
| **autoencoder** | convolutional autoencoder trained to rebuild *normal* road pixels; **reconstruction error = anomaly score** (583 k parameters) | not a chapter — see §6 |

**Protocol.** Grouped 70 / 15 / 15 split (39 / 9 / 12 images) so near-duplicate photos never straddle train and test; thresholds and
checkpoints chosen on **validation only**; **test used once**; metrics computed at **native resolution** (1280 × 720) against the
original masks; accuracy is *not* a headline metric because predicting "all background" already gives ≈ 90 %.

**Headline result.** Transfer learning is by far the best model (test F1 0.747 vs 0.464 and 0.202, see §9). The from-scratch CNN is far behind and the autoencoder is
barely better than chance at ranking pixels — which is exactly the limitation that motivated the more complex pipeline of the original paper.

---

## 2. Repository map and how to run it

```
RoadAnomaly_Project.ipynb   main notebook: exploration → training → evaluation → report (VS Code / Colab)
app.py                      Streamlit testing interface (upload an image, move the threshold, compare models)
roadanomaly/                reusable Python package
  config.py                   paths, seed, resolution, scene groups, threshold grid
  drive_data.py               Google Drive access (Drive for Desktop / public link / Colab / OAuth)
  data.py                     inventory, integrity checks, grouped split, tf.data pipeline, augmentation
  models.py                   baseline CNN, MobileNetV2 U-Net, autoencoder (+ scorer) and the custom losses
  training.py                 optimiser factory, compile, fit with checkpoint + early stopping, run bookkeeping
  metrics.py                  precision, recall, F1/Dice, IoU, AUROC, AP, FPR@95 (exact, pixel level)
  inference.py                experiment discovery, model loading, prediction, visualisation helpers
  experiments.py              chapter-3 studies: optimisers, regularisation, random / coarse-to-fine search
  report.py                   technical-report generator (reads every number back from disk)
outputs/                    runs/, experiments/, model_selection/<id>/ (checkpoints, metrics, figures, report)
technical_report/           report of the first Colab run (French) — kept for history
DEEPLEARING_FINALE.ipynb    first Colab notebook — kept for history
test_images/                two ready-to-use images (+ masks) for the Streamlit demo
DEEPLEARNING_class_course/  course material (not part of the code)
```

**Run it**

```powershell
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py                    # test the trained models (no dataset needed: upload an image)
python -m roadanomaly.experiments       # (optional) rerun the chapter-3 studies, about 40 min on CPU (cached results are skipped)
# notebook: open RoadAnomaly_Project.ipynb, choose the .venv kernel, Run All (about 25 min training + 40 min for the studies on CPU; see COMMANDS.txt)
```

Reproducibility: `keras.utils.set_random_seed(42)` before every run, the split is deterministic (seed 42 over declared scene groups),
all run configurations are saved next to the checkpoints (`*_run_config.json`). TensorFlow on native Windows is CPU-only (≥ 2.11), which is why runs are
slow-ish but still minutes, not hours, thanks to the tiny dataset. Exact numbers can differ in the third decimal between machines
(floating-point non-determinism in oneDNN / multithreaded reductions).

---

## 3. How the project follows the course

| Course material | What the course says | Where it is in the project |
|---|---|---|
| **Ch. 1 Introduction** | Artificial neuron = weighted sum (aggregation) + activation; perceptron update `W ← W + α(y − ŷ)X`; a purely linear network collapses to one layer, so non-linearities (sigmoid, tanh, ReLU) are needed; forward propagation, cost function, back-propagation (chain rule), gradient descent; timeline CNN 1990 → ImageNet 2012 → GAN 2014 …; conclusion lists RNN, CNN, GAN, **Autoencoder** | Every `Conv2D(…, activation="relu")` is a bank of neurons sharing weights; the output `sigmoid` squashes a score to [0, 1] like the course's sigmoid neuron; `model.fit` = forward → loss → backward (chain rule) → gradient step. Autoencoder → §6. |
| **Ch. 1 (team version)** | Limits of ANN (flattened input loses spatial structure, fixed size), CNN, RNN → LSTM → Bi-RNN, generative milestones (VAE 2013, GAN 2014, Transformers, Diffusion) | We use a CNN precisely because a *fully-connected* network would flatten 691 200 inputs and lose the geometry. The autoencoder is the ancestor of the VAE. |
| **Ch. 2 Loss functions** | MSE / MAE for regression; binary cross-entropy from the Bernoulli likelihood (`−Σ y log ŷ + (1−y) log(1−ŷ)`); categorical CE; worked chain-rule example `∂L/∂w = x(a − y)` | **BCE** per pixel for the two segmentation models (a pixel is a Bernoulli variable: anomaly / not); **MSE** for the autoencoder (a reconstruction is a regression). The Dice term is added because of the 90/10 imbalance (§7). |
| **Ch. 3 Hyper-parameters, regularisation, optimisation** | parameters vs hyper-parameters; learning rate, batch size, epochs; activations; over-fitting, bias–variance; **early stopping**, **dropout**, **L1 / L2**; batch / SGD / mini-batch; **momentum, RMSprop, Adam**; grid / random / **coarse-to-fine** search | Early stopping + best-checkpoint on every run; mini-batch of 4; Adam; data augmentation. All the other tools are **compared experimentally in notebook §7.4 / `experiments.py`** (optimisers, dropout, L2, batch-norm, random then coarse-to-fine search). |
| **Ch. 4 CNN** | 1D/2D/3D convolution, ReLU, **padding, stride**, pooling (max/min/avg), flatten, architectures LeNet-5, AlexNet, ZFNet, VGG, GoogLeNet, **ResNet** | `Conv2D(padding="same")`, `MaxPooling2D`, stride-2 convolutions (autoencoder). MobileNetV2 (a CNN architecture of the same family as the ones listed) is reused through transfer learning. No flatten/dense layer: the network is *fully convolutional* so the output is an image. |
| **Ch. 10 Sequence models (RNN, LSTM, GRU, BPTT)** | memory over time, vanishing / exploding gradients, gated cells | **Not applicable** and deliberately not used: each RoadAnomaly sample is one still photograph, there is no time axis. (See Q&A: "Why no RNN?".) A video extension could add a temporal model. |
| **Autoencoder** | only named in the ch.1 conclusion; no chapter | §6 explains it from the notions above (encoder, decoder, MSE, bottleneck). |

**Chapter-3 vocabulary applied to this project**

* *Parameters* (learned by gradient descent): the weights and biases of the convolution kernels (e.g. 46 689 in the baseline).
* *Hyper-parameters* (set by us): learning rate, batch size, number of epochs (bounded by early stopping), number of filters, kernel size, loss weights, dropout rate, L2 λ, the decision threshold.
* *One epoch* = one pass over the 39 training images = ⌈39/4⌉ = 10 gradient updates (iterations) with batch size 4.

---

## 4. Data

### 4.1 The RoadAnomaly dataset
* 60 images `frames/<name>.webp`, each with `frames/<name>.labels/labels_semantic.png`. Labels **{0 = background, 2 = anomaly}**; the loader *rejects* any other label.
* Families (from the file name): **animals 31, obstacles 15, cones 8, lost_tires 3, vehicle 3**.
* Integrity check (cell 3.2): 60 valid pairs, 0 errors, all 1280 × 720, every image contains an anomaly.
* Class imbalance: **9.85 %** anomaly pixels overall; per image median 6.6 %, min 0.37 % (a small bird) and max 42.3 % (a close-up animal). → huge scale variation.

### 4.2 Why a *grouped* split
Several photos show the same scene (3 lost-tyre shots, 5 cardboard-hedgehog shots, 4 rock shots, 2 Broadmoor shots). With a random image-level split
a near-copy of a test image would be in training: the score would measure memorisation, not generalisation (**data leakage**).
We declare these **scene groups** in `config.SCENE_GROUPS`; every other image is its own group. Groups are shuffled with seed 42 and cut 70 / 15 / 15 %
→ **train 39 / validation 9 / test 12 images**. `validate_split` checks that no image or group is shared and that the partitions cover the dataset exactly.

*Consequence to be aware of:* the five hedgehog photos form one group, so they all landed in **test** (5 of 12 test images = 42 %) and the model never saw a hedgehog. The test is therefore
harder and more "open-set" than a random split would be, and the *obstacles* family is dominated by this one object.

### 4.3 Preprocessing
* **Resize to 640 × 360** (half of native; keeps the 16:9 ratio; dimensions divisible by 8 for the pooling stages; we checked no annotation vanishes). Images: bilinear; masks: **nearest neighbour** (keeps labels binary); then labels 0/2 → 0/1.
* **Scale to [0, 1]** (`float32`). (MobileNetV2 expects [−1, 1]: the transfer model starts with `Rescaling(2, offset=−1)`.)
* **Augmentation (training only)**: horizontal flip applied jointly to image and mask, brightness ±0.10, contrast 0.85–1.15, saturation 0.85–1.15. These create "new" training examples without changing the label — a regulariser (chapter 3: more data fights over-fitting).
* **Evaluation at native resolution**: score maps are upsampled bilinearly to 1280 × 720 and compared with the original masks, so down-sampling errors are *included* in the score.

---

## 5. The three models in detail

All models: input `(360, 640, 3)` in [0, 1] → output `(360, 640, 1)` in [0, 1].

### 5.1 Baseline CNN (from scratch) — 46 689 parameters

| Layer | Output shape | Params | Role |
|---|---|---|---|
| `encoder_conv_1` Conv 3×3, 16, ReLU | 360×640×16 | 448 | edges / colours |
| `encoder_pool_1` MaxPool 2 | 180×320×16 | 0 | ½ resolution |
| `encoder_conv_2` Conv 3×3, 32, ReLU | 180×320×32 | 4 640 | textures |
| `encoder_pool_2` MaxPool 2 | 90×160×32 | 0 | ¼ resolution |
| `bottleneck` Conv 3×3, 64, ReLU | 90×160×64 | 18 496 | most abstract features |
| `decoder_upsampling_1` ×2 (nearest) | 180×320×64 | 0 | |
| `decoder_conv_1` Conv 3×3, 32, ReLU | 180×320×32 | 18 464 | |
| `decoder_upsampling_2` ×2 (nearest) | 360×640×32 | 0 | |
| `decoder_conv_2` Conv 3×3, 16, ReLU | 360×640×16 | 4 624 | |
| `anomaly_probability` Conv 1×1, 1, **sigmoid** | 360×640×1 | 17 | per-pixel probability |

*Parameter count of a convolution:* `k·k·C_in·C_out + C_out` — e.g. `3·3·3·16 + 16 = 448`. `padding="same"` keeps the spatial size (course ch. 4: padding preserves dimensions); pooling halves it (stride 2).
*Weights initialisation:* He-normal (suited to ReLU; chapter 3 mentions initialisation).

**Receptive field** (the area of the input one output pixel "sees"): conv3 → 3; pool → 4; conv3 → 8; pool → 10; conv3 → 18; (after the two up-samplings) conv3 → 22, conv3 → 24. **≈ 24 × 24 pixels on a 640 × 360 image.** The network judges each pixel from a tiny neighbourhood, with no notion of "this is a whole cow". That is the root cause of its false alarms and of its failure on large objects (§10).

### 5.2 Transfer learning: MobileNetV2 U-Net — 3 334 321 parameters (phase 1: 1 492 337 trainable / 1 841 984 frozen; after fine-tuning: 2 690 801 / 643 520)

* **Encoder** = MobileNetV2 (CNN architecture built from *inverted residual* blocks with depthwise-separable convolutions; light, ≈ 2.2 M parameters in the original) pre-trained on **ImageNet** (1.2 M images, 1 000 classes — many are animals). We tap five feature maps at strides 2, 4, 8, 16, 32.
* **Decoder** = four *decoder blocks*: up-sample ×2 → **concatenate the skip connection** from the encoder at that resolution → two Conv 3×3 + ReLU (128, 64, 32, 16 filters), then a final up-sample, a Conv 3×3 (8) and a 1×1 sigmoid. Input is zero-padded from 360 to 384 rows (a multiple of 32) and the padding is cropped at the output.
* **Why it works:** transfer learning (generic edges → textures → object parts learnt on 1.2 M images) replaces features that 39 images could never teach; **skip connections** (U-Net, Ronneberger 2015) give the decoder the fine spatial detail lost by down-sampling → sharp contours.
* **Two phases (chapter 3: learning-rate control):** (1) *frozen encoder*, only the decoder is trained (lr 3·10⁻⁴); (2) *fine-tuning*: the top 30 layers of the encoder are unfrozen and trained with a 10× smaller lr (10⁻⁴) for ≤ 15 epochs; the fine-tuned checkpoint is kept **only if validation loss improves**. BatchNorm layers stay in inference mode (frozen statistics), the standard recipe when fine-tuning on tiny data.

### 5.3 Autoencoder — 583 499 parameters (see §6)

---

## 6. The autoencoder explained from scratch

*(The course has no autoencoder chapter — only the word in the chapter-1 conclusion — so be ready to explain it from first principles.)*

### 6.1 Definition
An **autoencoder** is a neural network `x → z → x̂` trained so that the output `x̂` equals the input `x`.
* **Encoder** `f`: compresses the input into a **latent code** `z = f(x)` (a "bottleneck", much smaller than `x`).
* **Decoder** `g`: rebuilds `x̂ = g(z)`.
* **Loss**: a reconstruction loss, usually MSE `L = (1/N) Σ (x − x̂)²` (chapter 2, regression loss).
* **Training**: no labels are needed for the reconstruction target (the target *is* the input) → "self-supervised". Optimisation is the usual forward / back-propagation / Adam.

Without a bottleneck the network could learn the identity function and learn nothing. The **bottleneck forces it to keep only the most informative structure** of the data it was trained on. A linear autoencoder with MSE learns the same subspace as **PCA**; with ReLU/convolutions it learns a non-linear generalisation. Variants: denoising autoencoder, sparse autoencoder, **VAE** (variational, 2013, in the course timeline), GAN-based resynthesis.

### 6.2 Why it can detect anomalies
The decoder only knows how to produce what the encoder's code can describe, and the code was fitted to *normal* data. An unusual object (a cow on the road) is not in that vocabulary, so it is **reconstructed badly** → large pixel error exactly at the anomaly:

`score(p) = clip( avg9×9( mean_c (x_c(p) − x̂_c(p))² ) / s , 0, 1 )`

* mean over the 3 colour channels; `avg9×9` a 9 × 9 average pool (removes isolated noisy pixels); `s` = 99th percentile of that smoothed error over background pixels of the **training** images (`s = 0.0259` in the final run), so score 1 means "as badly reconstructed as the worst 1 % of normal pixels". The scorer is built from standard Keras layers, so it is saved as a normal `.keras` file.

This is a light version of **resynthesis** (Lis, Nakka, Fua, Salzmann, ICCV 2019): they re-synthesise the image from its semantic segmentation with a GAN and compare it with the input through a learned discrepancy network.

### 6.3 Our architecture
```
Input 360×640×3
 Conv3×3  32 (ReLU)   → 360×640×32     Conv3×3 s2 32 → 180×320×32    ┐
 Conv3×3  64 (ReLU)   → 180×320×64     Conv3×3 s2 64 →  90×160×64    ├ encoder (stride-2 convolutions instead of pooling)
 Conv3×3 128 (ReLU)   →  90×160×128    Conv3×3 s2 128 → 45×80×128    ┘
 Conv1×1   8 (ReLU)   →  45×80×8     ← latent code, 28 800 numbers = 24× smaller than the 691 200-value input
 ConvTranspose3×3 s2 128 + Conv3×3 128 → 90×160×128                  ┐
 ConvTranspose3×3 s2  64 + Conv3×3  64 → 180×320×64                  ├ decoder (transposed convolutions up-sample ×2)
 ConvTranspose3×3 s2  32 + Conv3×3  32 → 360×640×32                  ┘
 Conv1×1 3 (sigmoid)  → 360×640×3     reconstruction in [0, 1]
```
**No skip connection on purpose:** a skip would hand the decoder the input and the error would vanish everywhere, anomalies included.

### 6.4 Training details specific to anomaly detection
* **Masked MSE**: the loss is computed on **background pixels only** (`weights = 1 − anomaly_mask`). The network is never rewarded for reproducing animals or rocks, so it should reconstruct them poorly at test time. *This uses the ground-truth masks during training*, so the approach is **weakly supervised** (a fully unsupervised version would train on anomaly-free images, which RoadAnomaly does not contain: every image has an anomaly).
* Same optimiser (Adam, lr 10⁻³), augmentation, early stopping as the other models; ≤ 60 epochs.
* The threshold on the normalised score is picked on validation like for the other models.

### 6.5 Why it performs poorly here (and why that is an honest, expected result)
1. **Textures are hard to reconstruct.** A small (583 k parameters) autoencoder with a narrow bottleneck blurs fine detail: leaves, grass, road markings, edges of the horizon all give high error → many **false alarms** while the anomaly itself, if smooth, can be reconstructed fine → **misses**.
2. **No semantics.** It measures "unusual appearance", not "obstacle". Shadows, signs and parked cars are unusual too.
3. **Tiny training set**: 39 images do not describe "normal roads" well; a bigger driving dataset (Cityscapes) is what the paper uses.
4. **Score calibration.** After normalising by the 99th percentile, most pixels score far below 0.05; the best threshold on the validation grid sits at the lowest value (0.05), i.e. the grid is too coarse for this model. A finer grid near 0 would be a cheap improvement.

→ The autoencoder is included as the *complementary, reconstruction-based family*; its weakness is a **finding**, consistent with the literature.

---

## 7. Losses, optimisers, regularisation, hyper-parameters

### 7.1 Losses (chapter 2)
* **Binary cross-entropy** (per pixel, averaged): `BCE = −[y log ŷ + (1−y) log(1−ŷ)]`. Comes from the Bernoulli likelihood (course slides 5–6): minimising BCE = maximising the likelihood of the labels. Gradient through a sigmoid output: `∂L/∂z = ŷ − y` (the slide's `x(a − y)` result) — it does **not** vanish when the unit saturates, unlike MSE with a sigmoid.
* **Dice loss**: `1 − (2·Σ y ŷ + 1) / (Σ y + Σ ŷ + 1)` (computed per image). It measures **overlap**, so it is insensitive to how many background pixels there are.
* **Why BCE + Dice** (`bce_dice`, used by baseline and transfer): with 90 % background, plain BCE can be driven low by under-predicting anomalies; the Dice term optimises directly the quantity we are evaluated on and counters the imbalance.
* **Masked MSE** for the autoencoder (§6).

### 7.2 Activations (chapter 3)
**ReLU** in every hidden layer (cheap, does not saturate for positive inputs, mitigates vanishing gradients; chapter 3: "avoids and rectifies the vanishing gradient problem"); **sigmoid** at the single-channel output (probability, binary classification — chapter 3: "Sigmoid … for binary classification"). Softmax would be for multi-class.

### 7.3 Optimiser and learning rate (chapter 3)
**Adam** (adaptive moments: momentum on the gradient + RMSprop-style scaling by the running squared gradient, with bias correction), default `β₁ = 0.9, β₂ = 0.999`, lr `10⁻³` (baseline, autoencoder) or `3·10⁻⁴` (transfer: the decoder must adapt to pre-trained features without destroying them) and `10⁻⁴` for fine-tuning. **Mini-batch** gradient descent with batch size 4 (course: between batch GD and SGD; limited by memory: one 640×360 image with 32-filter feature maps is already several MB per layer).
The notebook §7.4 **compares SGD, SGD + momentum, RMSprop and Adam** experimentally.

### 7.4 Regularisation tools used (chapter 3)
| Tool | Used? | How / why |
|---|---|---|
| Early stopping | **yes, everywhere** | `EarlyStopping(monitor="val_loss", patience=7–10, restore_best_weights=True)` + `ModelCheckpoint(save_best_only)`; stops when validation loss stops improving (bias–variance trade-off) |
| More data / augmentation | **yes** | flips + photometric jitter on the fly |
| Transfer learning | **yes** | pre-trained encoder = strong prior (an implicit regulariser) |
| Dropout | studied in §7.4 | spatial dropout 0.4 (course: 40–50 % in CNNs) on the baseline |
| L2 / weight decay | studied in §7.4 | λ = 10⁻⁴ on the kernels; adds `λ‖W‖²` to the loss, the update becomes `W ← (1 − αλ)W − α∂L/∂W` |
| Batch normalisation | studied in §7.4 | Conv → BN → ReLU |
| L1 | not used | would drive weights to exactly 0 (sparsity); not a goal here |

### 7.5 Hyper-parameter search (chapter 3)
The chapter lists grid, random, coarse-to-fine, Bayesian, genetic and manual search. We used **manual** search for the main models (a few informed values) and ran
**random search followed by coarse-to-fine** on the baseline (§7.4 of the notebook): 5 random trials (learning rate log-uniform in [10⁻⁴, 10⁻²], batch size ∈ {2, 4, 8}), then 3 refined trials around the best learning rate. Learning rates are sampled on a **log scale** because the effect of lr is multiplicative.

---

## 8. Evaluation protocol and metrics

**Protocol (frozen before looking at the test set).** (1) predict validation images, upsample to native resolution; (2) per model, sweep thresholds 0.05 … 0.95 (step 0.05) and keep the one maximising **pooled pixel F1** (ties: higher recall, then closest to 0.5); (3) **primary model** = best validation F1; (4) the **test set is evaluated once** with those thresholds. Nothing is tuned on test.

**Confusion counts per pixel**: TP (anomaly predicted anomaly), FP (false alarm), FN (missed), TN. "Pooled" = sum the counts over all images, then compute the metric (so large objects weigh more); per-image metrics show the spread.

| Metric | Formula | Meaning here |
|---|---|---|
| Precision | TP / (TP + FP) | of the pixels flagged, how many are truly obstacle (low = many false alarms) |
| Recall | TP / (TP + FN) | of the true obstacle pixels, how many are found (low = missed obstacle — the dangerous error) |
| F1 = Dice | 2TP / (2TP + FP + FN) | harmonic mean of precision and recall; F1 and Dice are **identical** for binary masks |
| IoU (Jaccard) | TP / (TP + FP + FN) | overlap of prediction and truth; IoU = F1 / (2 − F1) |
| Accuracy | (TP + TN) / all | **not informative**: "all background" scores ≈ 90 % |
| **AUROC** | area under TPR vs FPR over *all* thresholds | probability that a random anomaly pixel gets a higher score than a random background pixel; 0.5 = chance, 1 = perfect; threshold-free |
| **AP** | area under the precision–recall curve | better than AUROC with strong imbalance; chance level = anomaly fraction ≈ 0.10 |
| **FPR @ 95 % TPR** | false-positive rate when 95 % of anomaly pixels are found | the metric of Lis et al.; lower is better |

AUROC, AP and FPR@95 are computed **exactly over all pixels** (ties in the scores are grouped so the result does not depend on pixel order); thresholded metrics need the threshold chosen on validation.

---

## 9. Results

All numbers come from the run executed in `RoadAnomaly_Project.ipynb` (experiment `outputs/model_selection/20261107T094734_134561Z`; the same numbers are in
`final_evaluation/test_results.json` and in the generated report). Test set = 12 images, used once.

### 9.1 Training summary

| Model | Trainable / frozen params | Loss | Learning rate | Epochs (best) | Time (CPU) | Threshold (validation) |
|---|---|---|---|---|---|---|
| baseline | 46 689 / 0 | BCE + Dice | 1e-3 | 34 (24) | 2.2 min | 0.20 |
| transfer | 2 690 801 / 643 520 *(after fine-tuning; phase 1 had 1 492 337 / 1 841 984)* | BCE + Dice | 3e-4, then 1e-4 | 23 frozen (13) + 15 fine-tune (best overall at epoch 29) | 3.3 + 2.5 min | 0.35 |
| autoencoder | 583 499 / 3 | masked MSE | 1e-3 | 60 = cap (58) | 14.1 min | 0.05 *(grid edge)* |

The autoencoder used all 60 epochs (still improving slowly); its error scale `s` (99th percentile of background error) is 0.0259.

### 9.2 Pooled pixel metrics

| Model | Set | Precision | Recall | F1 = Dice | IoU | AUROC | AP | FPR@95 |
|---|---|---|---|---|---|---|---|---|
| baseline | validation | 0.336 | 0.622 | 0.436 | 0.279 | 0.855 | 0.388 | – |
| baseline | **test** | 0.347 | 0.697 | **0.464** | 0.302 | 0.890 | 0.432 | 0.316 |
| transfer | validation | 0.904 | 0.859 | 0.881 | 0.787 | 0.993 | 0.949 | – |
| transfer | **test** | 0.811 | 0.692 | **0.747** | 0.596 | 0.952 | 0.817 | 0.258 |
| autoencoder | validation | 0.105 | 0.714 | 0.184 | 0.101 | 0.586 | 0.100 | – |
| autoencoder | **test** | 0.116 | 0.767 | **0.202** | 0.112 | 0.663 | 0.241 | 0.878 |

Primary model chosen on validation (best F1): **transfer** (also best on test). Reference levels: AUROC of a random scorer = 0.5; AP of a random scorer = anomaly fraction ≈ 0.10.

### 9.3 Test results per object family and per image (F1)

| Family (test images) | baseline | transfer | autoencoder |
|---|---|---|---|
| animals (6) | 0.484 | **0.889** | 0.274 |
| cones (1) | 0.208 | **0.613** | 0.048 |
| obstacles (5, all cardboard hedgehogs) | 0.451 | 0.433 | 0.132 |

Per-image F1 on test — transfer: mean 0.706, median 0.748, min 0.225 (`obstacles07_czech_hedgehog`), max 0.933 (peccary); baseline: mean 0.458, min 0.115 (pheasant), max 0.826; autoencoder: mean 0.182, min 0.034, max 0.500.
The cones family has **one** image, so its score is anecdotal.

### 9.4 Course chapter-3 studies (baseline CNN, validation only; notebook §7.4)

| Study | Best | Worst | Take-away |
|---|---|---|---|
| Optimiser | Adam 0.465 | SGD 0.391 (RMSprop 0.446, momentum 0.425) | adaptive methods converge faster; SGD hit the 40-epoch cap |
| Regularisation | Adam reference 0.465–0.482, L2 0.480 | dropout 0.4: 0.177 (failed), batch-norm 0.400 | baseline under-fits; BN needs larger batches; 0.4 dropout too strong for 46 k params |
| Search (random → coarse-to-fine) | lr 1.6e-3, batch 8: **0.506** | lr 1e-4–2e-3, batch 2–4: 0.44–0.48 | tuning gains ≤ 0.04; far below transfer (0.88 on validation) |

Noise level: two identical Adam runs gave 0.465 and 0.482 → differences < 0.02 F1 mean nothing.

**Reading the studies.**

* **Optimisers.** Adam (0.465) > RMSprop (0.446) > SGD + momentum (0.425) > plain SGD (0.391). The adaptive methods converge much faster at the same budget (Adam stopped after 21 epochs and 85 s; plain SGD used all 40 epochs, 254 s, and its best epoch was 38, i.e. it was still improving). This is the behaviour described in chapter 3: momentum damps the oscillations of SGD, RMSprop / Adam rescale the step per parameter. A longer or better-tuned SGD run could close part of the gap; it was not tested.
* **Regularisation.** Two identical Adam runs gave 0.465 and 0.482, so differences below about 0.02 F1 are noise. L2 (λ = 1e-4) gave 0.480: no measurable effect. In every run except batch-norm the validation loss is *lower* than the training loss (augmented images are harder than clean ones): the 46 k-parameter baseline **under-fits** rather than over-fits, so regularisation has little to correct. Batch-norm was worse (0.400; 40 epochs, validation loss above training loss): with batch size 4 the batch statistics are very noisy, and BN needs larger batches. Dropout 0.4 (the 40–50 % rule of thumb for CNNs) **failed to train** (F1 0.177, best epoch 1): a rate meant for large networks removes too much of a network with only 16–64 channels. A rate of 0.1–0.2 was not tested.
* **Hyper-parameter search.** The best random trial (learning rate 1.6e-3, batch size 8) reached 0.506; the three finer trials around it did not improve (best 0.505), so the search had already reached its plateau. Tuning improved the baseline by only 0.02-0.04 F1 over the default (0.465-0.482), and the tuned baseline stays far below the transfer model (validation F1 0.88). Lesson: **architecture and pre-training matter far more than hyper-parameter tuning here**. The tuned values were not used for the final baseline, to keep the three-model comparison on one fixed protocol.

### 9.5 Figures to show (all in `outputs/model_selection/20261107T094734_134561Z/` and in the notebook)
`validation_threshold_curves.png` (why each threshold) · `final_evaluation/test_metrics_comparison.png` · `roc_pr_curves.png` · `analysis/animals12_…_comparison.png` (success) ·
`analysis/obstacles07_…_comparison.png` (failure) · `analysis/autoencoder_reconstructions.png` (why the autoencoder fails) · `analysis/per_image_f1.png` · `outputs/experiments/summary.png` (chapter-3 studies) · learning curves in `outputs/runs/<model>/<run>/learning_curves.png`.

---

## 10. Critical analysis

**Transfer learning is the only model that works well, and it works for the right reason.** The pre-trained MobileNetV2 U-Net reaches test F1 0.747 / IoU 0.596 / AUROC 0.952, against 0.464 / 0.302 / 0.890 for the from-scratch CNN and 0.202 / 0.112 / 0.663 for the autoencoder (chance level for AP is about 0.10; the transfer model reaches 0.82). On animals, its strongest family, F1 is 0.889: the ImageNet encoder already knows animals, and the skip connections give clean contours (Figure 5, peccary: almost no false alarms, thin one-pixel border errors only).

**The baseline fails because of its receptive field and lack of prior knowledge, not because of over-fitting.** Its receptive field is about 24 x 24 pixels, so it judges each pixel from a small patch, with no notion of "a whole animal". It lights up high-contrast texture (the bush in the peccary image is almost entirely a false alarm) and only partly fills large objects (blue holes inside the animal). It needs a low threshold (0.20) to reach recall 0.70, at the price of precision 0.35. The chapter-3 studies confirm that it under-fits: regularisation does not help, and tuning the learning rate and batch size gains at most 0.04 F1.

**The autoencoder is close to useless alone, and the reconstructions explain why.** Its reconstruction error follows image *texture*, not *anomaly*: the bush and the shadow edge have a large error, while the dark, smooth, textureless cardboard hedgehog is reconstructed almost perfectly (error close to 0, Figure 6), so it is missed. A small autoencoder generalises to any simple shape and fails on any detailed texture. Its best validation threshold sits at the lowest value of the grid (0.05), where it flags most pixels (recall 0.77, precision 0.12, about the 10 % base rate): a degenerate operating point, and a sign that the grid is too coarse for its scores (they are normalised by a high percentile and are mostly below 0.05). This is the limitation that Lis et al. address with semantic segmentation, a GAN and a learned discrepancy network. It is also weakly supervised here: the ground-truth masks are used to exclude anomalies from the loss.

**Where the best model still fails: objects unlike anything seen.** The five cardboard-hedgehog photos form one scene group and were all placed in the test set, so the model never saw a hedgehog. The obstacles family reaches F1 0.433 and the worst test image, `obstacles07_czech_hedgehog`, 0.225 (recall 0.13: only the four arm tips are detected, Figure 7). The network has learnt "things that look like animals or cones" more than "any obstacle", which is the open-set problem of the original benchmark. Cones (1 test image, F1 0.613) and the small pheasant (F1 0.817, small object) are the other weak spots, and the contours are slightly blurred by the 640 x 360 working resolution.

**Validation-to-test gap.** The transfer model falls from 0.881 (validation) to 0.747 (test), whereas the baseline and the autoencoder do not drop. This is not classical over-fitting of the weights: the validation set (9 images) contains no hedgehog and the test set (12 images) is 42 % hedgehog photos. With sets this small, one hard scene moves the pooled scores by 0.1 or more, and two identical training runs differ by about 0.02 F1. No confidence intervals are reported; treat differences below 0.05 as not significant, while the gaps between the three models (0.28 to 0.54 F1) are far larger than that noise.

---

## 11. Limitations and what we would do next

* **Tiny data.** 60 images, 39 for training. Scores and thresholds have high variance; no confidence intervals; single seed. *Next:* group-aware k-fold cross-validation and several seeds → mean ± std.
* **Test = 12 images, 42 % one unseen object type.** The "obstacles" family is really "hedgehogs". *Next:* different group-to-split assignments (cross-validation) so that every group is tested once.
* **RoadAnomaly is a test benchmark.** Training on 39 of its images is a pedagogical protocol; it is not the open-set setting of the paper (anomalies never seen in training), and the numbers are not comparable with published results.
* **Autoencoder.** Weakly supervised (masks used in the loss), poor texture handling, threshold grid too coarse near 0. *Next:* finer grid; deeper autoencoder or perceptual loss; full resynthesis pipeline (segmenter → GAN → discrepancy network).
* **Resolution.** 640 × 360 blurs contours and loses small objects. *Next:* higher resolution / multi-scale inference.
* **Pre-training on the right domain.** ImageNet knows animals better than cardboard barriers. *Next:* Cityscapes pre-training + synthetic anomaly pasting (standard in the literature).
* **Hyper-parameter studies.** Only on the baseline, one seed, and dropout was tested at a single (too high) rate. *Next:* dropout 0.1–0.2, larger batches for BN, tuned transfer model.
* **Safety.** A missed obstacle (false negative) is more dangerous than a false alarm; the threshold should be set from an application-level recall target, and a real system needs far more data and validation.

---

## 12. Cheat-sheet of numbers

* Data: 60 images, 1280×720 → 640×360; anomaly pixels 9.85 % (per image 0.37 %–42.3 %, median 6.6 %); families 31 animals / 15 obstacles / 8 cones / 3 lost tyres / 3 vehicle.
* Split (seed 42, grouped): 39 train / 9 validation / 12 test. Test: 6 animals, 1 cone, 5 hedgehogs.
* Parameters: baseline 46 689 · transfer 3 334 321 (2 690 801 trainable after fine-tuning) · autoencoder 583 499. Baseline receptive field ≈ 24 px. Autoencoder code 45×80×8 = 28 800 values vs 691 200 (24× smaller).
* Training: Adam, batch 4, early stopping (patience 7–10, best weights restored), augmentation (flip, brightness, contrast, saturation), seed 42, CPU only.
* **Test F1 / IoU / AUROC / AP:** transfer **0.747 / 0.596 / 0.952 / 0.817** · baseline 0.464 / 0.302 / 0.890 / 0.432 · autoencoder 0.202 / 0.112 / 0.663 / 0.241.
* Validation F1: transfer 0.881 · baseline 0.436 · autoencoder 0.184. Thresholds: transfer 0.35 · baseline 0.20 · autoencoder 0.05.
* Chapter-3 studies (validation F1, baseline): Adam 0.465 > RMSprop 0.446 > momentum 0.425 > SGD 0.391; L2 0.480; BN 0.400; dropout 0.4 0.177; best searched config 0.506 (lr 1.6e-3, batch 8). Noise ≈ 0.02.
* Worst test image: `obstacles07_czech_hedgehog` (transfer F1 0.225). Best: peccary (0.933).

---

## 13. Questions on the course concepts (with answers)

**Chapter 1 — Introduction**

1. *What is deep learning versus machine learning?* ML = a program that learns a task from examples by minimising an error with an optimisation algorithm (course: fit `f(x) = ax + b` by minimising the distance to the data). DL = ML with **many-layer neural networks** that learn their own features from raw data (here: pixels) instead of hand-made features.
2. *What is an artificial neuron?* `a = φ(Σ wᵢxᵢ + b)`: an **aggregation** (weighted sum) followed by an **activation**. A convolution filter is a neuron applied at every position with the same weights.
3. *What is the perceptron rule?* `W ← W + α(y − ŷ)X`: the weights are reinforced when the input and the expected output are active together (Hebb). It is a **linear** model, so it cannot separate non-linear data (XOR).
4. *Why do we need non-linear activations?* A stack of linear layers is equivalent to one linear layer (course slide "limits of linear neurons"); hidden layers only add power if followed by a non-linearity (ReLU, tanh, sigmoid).
5. *Forward propagation, backward propagation, gradient descent?* Forward: compute and store all intermediate outputs from input to output. Cost: compare with the target. Backward: apply the **chain rule** layer by layer to get ∂L/∂(W, b). Gradient descent: `W ← W − α ∂L/∂W`. `model.fit` does exactly this in each iteration.
6. *Why did deep learning take off after 2012?* Big data, GPUs (parallelisable), better software/techniques (ReLU, dropout, frameworks) — AlexNet won ImageNet in 2012. In our project: ImageNet pre-training is what makes the transfer model good.

**Chapter 2 — Loss functions**

7. *MSE vs BCE?* MSE `(1/N)Σ(y − ŷ)²` for regression; **BCE** `−(1/N)Σ[y log ŷ + (1−y)log(1−ŷ)]` for binary classification, derived from the Bernoulli likelihood: maximising the likelihood = minimising BCE. We use BCE per pixel (anomaly yes/no) and MSE for the autoencoder (a reconstruction is a real-valued regression).
8. *Derive ∂L/∂w for a sigmoid neuron with BCE.* `∂L/∂a = (a − y)/(a(1 − a))`, `∂a/∂y = a(1 − a)`, `∂y/∂w = x` ⇒ `∂L/∂w = x(a − y)`, `∂L/∂b = a − y` (course slides 8–9; numerical example: w₁ = 0.29 → 0.38 with α = 0.2).
9. *Why not accuracy?* 90 % of pixels are background: a model that outputs "background" everywhere gets ≈ 90 % accuracy and finds nothing.
10. *Why add a Dice term?* Class imbalance. Dice measures overlap and ignores true negatives; combined with BCE it gives both a smooth per-pixel gradient (BCE) and the right objective (Dice).

**Chapter 3 — Hyper-parameters, regularisation, optimisation**

11. *Parameters vs hyper-parameters?* Parameters (W, b) are learned; hyper-parameters (learning rate, batch size, epochs, layers, units, dropout rate, λ…) are set before training. Listed for each model in the `*_run_config.json` files.
12. *What happens with a too large / too small learning rate?* Too large: the update jumps over the minimum and may diverge (course: lr = 0.8 oscillates); too small: very slow convergence (lr = 0.05). Hence a smaller lr (3·10⁻⁴, then 10⁻⁴) for the pre-trained transfer model.
13. *Batch, mini-batch, SGD?* Batch uses all samples per update (smooth but costly, 1 update/epoch); SGD uses one sample (noisy, m updates/epoch); **mini-batch** uses 1 < b < m (compromise). We use b = 4 → 10 updates per epoch on 39 images.
14. *What is overfitting? How is it seen?* The model fits the training data (incl. noise) and fails on new data: training loss keeps falling while **validation loss rises** (high variance). Seen in the learning curves; counter-measures: more data/augmentation, early stopping, dropout, L2, simpler model, transfer learning.
15. *Early stopping?* Monitor validation loss; stop after `patience` epochs without improvement and restore the best weights (`restore_best_weights=True`). Used in every run.
16. *Dropout?* At each training step each neuron is dropped with probability p, forcing redundancy (an ensemble effect); off at inference. Rates 20–30 % for RNNs, 40–50 % for CNNs (course). We use **spatial** dropout (drops whole feature maps, better for convolutions since neighbouring pixels are correlated) in the experiment.
17. *L1 vs L2?* L2 adds `(λ/2)‖W‖²` → weight decay `W ← (1 − αλ)W − α∂C₀/∂W`, small weights but not zero; L1 adds `λ‖W‖₁` → sparsity, many weights exactly 0.
18. *Momentum, RMSprop, Adam?* Momentum keeps an exponential average of past gradients (damps oscillations, accelerates); RMSprop divides the step by the root of the running average of squared gradients (per-parameter adaptive lr); **Adam** = momentum + RMSprop with bias correction (β₁ = 0.9, β₂ = 0.999). Compared in §7.4 / `experiments.py`.
19. *Batch normalisation?* Normalises each channel of a layer's activations over the mini-batch (then a learned scale and shift): stabilises and speeds training, slight regularising effect. Option `batch_norm=True` in the baseline; in the pre-trained MobileNetV2 BN statistics are kept frozen.
20. *Which hyper-parameter search methods exist?* Grid, random, coarse-to-fine, Bayesian, genetic, manual. Random search beats grid when only a few hyper-parameters matter; coarse-to-fine = random search, then a smaller region around the best. We did random → coarse-to-fine on lr and batch size.
21. *Sigmoid, tanh, ReLU, Leaky ReLU, softmax?* Sigmoid ∈ (0,1), not zero-centred, saturates (vanishing gradient); tanh ∈ (−1,1), zero-centred; ReLU `max(0, x)` cheap, no saturation for x > 0 but "dying ReLU" possible; Leaky ReLU keeps a small slope for x < 0; softmax gives class probabilities for multi-class. Ours: ReLU hidden, sigmoid output (binary).
22. *Why is the validation set separate from the test set?* Validation is used to *make decisions* (checkpoint, threshold, model choice) so it becomes optimistic; test must stay untouched to give an unbiased estimate. Our validation F1 is indeed higher than test F1 for the transfer model.

**Chapter 4 — CNN**

23. *Why CNN rather than a fully-connected network for images?* A dense network flattens the image (692 k inputs → huge parameter count, no spatial structure, fixed size). Convolution shares a small kernel across positions: few parameters, translation-equivariant, works at any size (fully convolutional).
24. *Output size of a convolution?* `⌊(n + 2p − k)/s⌋ + 1`; with `padding="same"` and stride 1 the size is unchanged; stride 2 halves it. Pooling `k = 2, s = 2` halves it.
25. *Parameters of a conv layer?* `k·k·C_in·C_out + C_out` (e.g. 448 for 3×3, 3→16).
26. *Padding, stride, pooling?* Padding preserves borders/size; stride is the step of the filter (larger stride → smaller map); pooling (max/min/average) down-samples and brings small translation invariance.
27. *Why ReLU after convolution?* A convolution is linear; ReLU adds the non-linearity, sparsity, cheap derivative, faster convergence.
28. *Name CNN architectures.* LeNet-5 (1998, 2 conv + 3 FC, tanh), AlexNet (2012, ImageNet, GPUs, ReLU), ZFNet (2013), GoogLeNet/Inception (2014), VGG (2014, small 3×3 stacks), ResNet (2015, **skip connections** — same idea as our U-Net skips, to ease gradient flow). We reuse **MobileNetV2** (efficient depthwise-separable convs).
29. *What is a 1D / 2D / 3D convolution?* 1D for sequences/signals, 2D for images (our case), 3D for volumes/video.
30. *Why is a CNN a "black box"?* Weights are random-initialised and learned; interpretability is an open research topic. We offer *error maps* and score heat-maps as visual diagnostics.

**Chapter 10 — Sequence models**

31. *Why no RNN/LSTM in this project?* An RNN models **order in time**; our samples are independent still images. Convolutions already exploit the 2D spatial structure. An RNN/ConvLSTM would be relevant if we processed **video** (frame t informs frame t+1), e.g. tracking an obstacle approaching.
32. *RNN limits and fixes?* Vanishing/exploding gradients through BPTT; exploding → gradient clipping; vanishing → ReLU, good initialisation, truncated BPTT, **gated cells (LSTM, GRU)**. (Not used here, but know them.)
33. *CNN limits vs RNN (course)?* CNNs need a fixed input size (in the classic dense-head form), have no memory and are order-invariant — fine for images, not for sequences.

**Autoencoder (outside the chapters)**

34. *What is an autoencoder and how is it trained?* See §6: encoder → bottleneck → decoder, target = input, MSE loss, Adam. Self-supervised (no external labels).
35. *Why does the bottleneck matter?* Without it the network learns the identity. A narrow code forces a compact summary of the **training distribution**; unseen content is reconstructed poorly.
36. *Autoencoder vs PCA?* A linear autoencoder with MSE spans the same subspace as PCA; non-linear (ReLU, convolutions) autoencoders capture curved manifolds.
37. *Autoencoder vs VAE vs GAN?* VAE (2013) makes the code a probability distribution (regularised with a KL term) so it can *generate*; GAN (2014) trains a generator against a discriminator; Lis et al. use a GAN to *re-synthesise* the road from semantic labels. Ours is a plain deterministic autoencoder.
38. *Other uses of autoencoders?* Denoising, dimensionality reduction, pre-training, anomaly detection (here), generation (VAE).

---

## 14. Questions on the project (with answers)

**Problem and data**

1. *Why pixel-level segmentation rather than image classification or box detection?* The obstacle's exact extent matters for driving (free space), the dataset provides pixel masks, and a per-pixel score lets us use threshold-free metrics (AUROC/AP). Classification would say "something is there" but not where; every image here contains an anomaly anyway.
2. *Why is this problem hard?* Anomalies are rare, varied and never seen by a normal driving dataset (open-set); here they cover ≈ 10 % of pixels and range from 0.4 % to 42 % of an image; there are only 60 images.
3. *What is RoadAnomaly?* A benchmark of 60 road photos with unexpected objects, from Lis et al. (ICCV 2019). It is meant for **testing** models trained on other data; we split it ourselves for a pedagogical supervised protocol and say so.
4. *Why resize to 640 × 360?* Half resolution keeps the 16:9 ratio, divisible by 8 for the pooling stages, fits CPU memory/time (a full-size feature map of 32 filters is 118 MB per image in float32), and we verified that no annotation vanished. Evaluation is at native resolution.
5. *Why did you not use vertical flips or rotations?* Road scenes have a fixed orientation (sky up, road down); a flipped image is not a realistic sample. Horizontal flips are realistic. Photometric jitter simulates lighting changes.

**Split and evaluation**

6. *What is data leakage and where could it occur?* Information from the test set reaching training. Here: near-duplicate photos of the same scene (3 tyres, 5 hedgehogs, 4 rocks, 2 Broadmoor) split across partitions. We group them (`SCENE_GROUPS`) and `validate_split` checks it. Also thresholds are tuned on validation, never on test.
7. *Why 70/15/15 and not k-fold cross-validation?* Training three models several times on CPU is costly, and k-fold with groups needs care; we accept the high variance of a single split and **say so**. Cross-validation is the first improvement listed.
8. *Why is the validation set separate from the test set?* Validation is used for decisions (best epoch, threshold, primary model) so its score is optimistic: transfer F1 0.881 on validation vs 0.747 on test shows it.
9. *Why not report accuracy?* "All background" scores ≈ 90 % accuracy. We report precision, recall, F1/Dice, IoU, and threshold-free AUROC/AP/FPR@95.
10. *F1 vs Dice vs IoU?* F1 = Dice = 2TP/(2TP + FP + FN) for binary masks; IoU = TP/(TP + FP + FN) = F1/(2 − F1), always lower. We show both because papers use either.
11. *What do precision and recall mean for a driving system?* Recall = fraction of the obstacle found (a miss is dangerous); precision = fraction of alarms that are real (false alarms cause needless braking). The threshold trades one for the other; we pick the F1-optimal one but a deployed system would favour recall.
12. *What is AUROC? Why also AP?* AUROC = probability that a random anomaly pixel scores higher than a random background pixel (0.5 = chance). With 90 % negatives, AP (area under precision–recall) is more demanding: random = 0.10. FPR@95 = false-positive rate needed to find 95 % of anomaly pixels.
13. *How was the threshold chosen?* Sweep 0.05–0.95 on validation, keep the best pooled F1 (ties: higher recall, closest to 0.5); then frozen. Saved in `validation_selection.json` *before* the test is evaluated.
14. *What does "pooled" mean?* Counts TP/FP/FN/TN are summed over all images before computing the metric; large objects weigh more. Per-image metrics show the spread (transfer: 0.225 to 0.933).
15. *Are the differences between models significant?* The gaps between models (0.28–0.54 F1) are far larger than the noise we measured (≈ 0.02 between identical runs); differences below ≈ 0.05 would not be.

**Models and training**

16. *Why three models?* Baseline = what can be learnt from scratch (lower reference); transfer = standard practical solution for small data; autoencoder = a different family (reconstruction-based, the idea of the reference paper) to compare.
17. *What is transfer learning and why does it help here?* Re-using a network trained on a large dataset (ImageNet, 1.2 M images). Its early layers detect edges/textures, later ones object parts; 39 images could never teach that. Result: F1 0.747 vs 0.464 for the same task from scratch.
18. *Why MobileNetV2?* Small (≈ 2.2 M parameters in the original), fast on CPU, good ImageNet features; its internal feature maps at five strides are easy to tap for a U-Net. A ResNet or VGG would also work but is heavier.
19. *What is a U-Net and what are skip connections for?* Encoder–decoder with *concatenations* between encoder and decoder feature maps of the same resolution. Down-sampling loses precise location; skips bring it back → sharp mask contours. (Same idea as ResNet's skips in spirit: short paths for information and gradients.)
20. *Why freeze the encoder first, then fine-tune?* The decoder starts random: large random gradients would destroy good pre-trained weights. So train the decoder alone (lr 3e-4), then unfreeze the top 30 encoder layers with a 10× smaller learning rate (1e-4); keep the fine-tuned model only if validation loss improves (it did: 0.317 → 0.293).
21. *Why keep BatchNorm frozen when fine-tuning?* With 39 images and batch 4 the batch statistics are noisy; the pre-trained running statistics are more reliable.
22. *Why BCE + Dice?* BCE is the per-pixel likelihood loss (course ch. 2) but with 90 % background it can be minimised by predicting little; Dice optimises overlap (our metric) irrespective of the background size. Their sum gives smooth gradients and the right objective.
23. *Why Adam, lr 1e-3 / 3e-4 / 1e-4, batch 4?* Adam converges fast without hand-tuned schedules (our study: Adam beat SGD, momentum and RMSprop on the baseline); lower learning rates for pre-trained weights; batch 4 is the memory limit on CPU and gives 10 updates/epoch.
24. *How many epochs? Why early stopping?* Up to 50–80 (baseline), 60 (transfer, autoencoder), but training stops when validation loss has not improved for 7–10 epochs and the best weights are restored: avoids over-fitting and saves time. Baseline stopped at 34 epochs (best 24).
25. *Is there over-fitting?* The baseline under-fits (validation loss below training loss, regularisation does not help). For the transfer model the validation→test drop (0.881 → 0.747) comes mostly from a distribution shift (hedgehogs only in test), not from memorisation; early stopping + augmentation + frozen pre-trained features limit over-fitting.
26. *Why is the baseline so much worse?* Receptive field ≈ 24 px: it sees textures, not objects (the bush lights up, the animal is only partly filled); only 46 k parameters learnt from 39 images; no prior knowledge. It needs a low threshold (0.20) so precision is 0.35.
27. *Why did dropout 0.4 fail on the baseline?* The 40–50 % rule is for large CNNs; with 16–64 channels, spatial dropout of 0.4 removes too much capacity, the model never learned (best epoch 1, F1 0.177). Lower rates (0.1–0.2) were not tested.
28. *Why does batch-norm not help?* Batch size 4 → noisy mean/variance estimates; BN is designed for larger batches.
29. *Random search or grid search? Why log-uniform?* Random search explores more distinct values per hyper-parameter when few of them matter; the learning rate acts multiplicatively, so we sample its exponent uniformly. Then **coarse-to-fine**: 3 more trials within a factor 2 of the best.

**Autoencoder**

30. *Is your autoencoder unsupervised?* Not fully: it is trained to reconstruct, with the background-only mask excluding anomaly pixels from the loss, so it uses the labels. A fully unsupervised version needs anomaly-free images, which RoadAnomaly does not provide.
31. *Why the bottleneck and no skip connection?* To forbid copying. With skips the reconstruction would be perfect everywhere and the error would be 0 on anomalies as well.
32. *Why does it perform poorly?* Reconstruction error follows **texture**: bushes, shadows, edges are hard to rebuild (false alarms), whereas a smooth dark object like the hedgehog is rebuilt well (miss). AUROC 0.66 only. Best threshold at the grid edge (0.05), where it flags ≈ 65 % of the pixels.
33. *Why use it then?* As a complementary, label-light family and an honest negative result: it shows why the paper needs semantics + GAN + learned comparison.
34. *How is the anomaly score computed?* `clip(avgpool9×9(mean_rgb (x − x̂)²)/s, 0, 1)`, `s` = 99th percentile of the background error on the training images (0.0259).

**Results and interpretation**

35. *Which model is the best and why?* Transfer: F1 0.747, IoU 0.596, AUROC 0.952, best precision, F1, IoU, AUROC and AP, and best on animals and cones; on obstacles it ties with the baseline (0.433 vs 0.451); the autoencoder (0.77) and the baseline (0.70) have a slightly higher recall (transfer 0.69) but only because they flag many more pixels.
36. *Where does the best model fail?* Hedgehogs (never seen): `obstacles07` recall 0.13, only the arm tips; the small pheasant has F1 0.817 (fine) but the baseline gets 0.115; the cone image 0.613. It learned "animal/cone-like things" more than "any obstacle".
37. *Why does the test F1 differ from the validation F1?* Tiny sets (9 / 12 images) and a distribution shift (hedgehogs only in test). One hard scene moves pooled scores by 0.1.
38. *How do your results compare with the paper?* Not comparable: the paper trains on other data (Cityscapes) and tests on all 60 images in an open-set setting; we train on 39 of them. We only share the metrics (AUROC, AP, FPR@95) and the resynthesis idea.
39. *What would you do with one more week?* Cross-validation with several seeds; Cityscapes pre-training + synthetic anomalies; higher resolution; a finer threshold grid and a stronger autoencoder; test dropout 0.1–0.2.
40. *Is this usable in a real car?* No: 60 images, no domain diversity, a miss rate of ≈ 31 % of obstacle pixels at the F1-optimal threshold. It is a study of methods, not a safety system.

**Implementation and reproducibility**

41. *How is the project reproducible?* Seed 42 everywhere, deterministic grouped split, saved run configs, thresholds frozen in JSON before the test, the report generated from files on disk, `COMMANDS.txt` with the exact commands. Residual non-determinism (~0.02 F1) comes from multithreaded floating-point operations.
42. *Where is the data and how is it loaded?* On Google Drive (professor's folder, with a fallback copy); `drive_data.ensure_dataset()` finds a local copy, Drive for Desktop, or downloads into `data/` (git-ignored). Integrity is checked against `frame_list.json` (60 images + 60 masks).
43. *What does the Streamlit app do?* Four tabs: *Test an image* (dataset image or upload, choose models, threshold slider, score heat-map, predicted mask, TP/FP/FN error map, per-image metrics, download), *Experiment results* (tables, curves, per-image and per-family scores), *Dataset explorer*, *About*. It loads the experiment folder found in `outputs/` (the newest complete one).
44. *What does the threshold slider do?* Converts the score map into a binary mask: lower = more recall, more false alarms; higher = the opposite. The default is the validation-optimal value.
45. *Why TensorFlow/Keras and CPU?* Keras gives pre-trained MobileNetV2 and a compact training API; TensorFlow ≥ 2.11 has no native-Windows GPU support, and the dataset is small enough for CPU (≈ 25 min for all three models).

**Live demo script (3 minutes)**
1. `streamlit run app.py` → *Test an image* → choose `test_images/test_easy_peccary.jpg` (+ its mask): select **transfer**, show the clean mask; add **baseline** and **autoencoder** to show false alarms and the heat-maps.
2. Load `test_images/test_hard_hedgehog.jpg`: show the transfer model missing most of the hedgehog; move the threshold slider to show the precision/recall trade-off.
3. *Experiment results* tab: the comparison table and per-image F1; say that test was used once with thresholds frozen on validation.

---

## 15. Honest list of weaknesses (say them before the professor finds them)

1. **Very small dataset and test set** (60 / 12 images), single seed, no confidence intervals; the test set is 42 % one unseen object type.
2. **The autoencoder is weak** (F1 0.20, AUROC 0.66), uses the masks during training, ends on the edge of the threshold grid (0.05) and was still improving at its 60-epoch cap.
3. **The course has no autoencoder chapter**; we explain it from the chapter-1 to -4 notions and cite the reference paper. There is no RNN because the data are still images.
4. **Chapter-3 studies are on the baseline only**, one seed per variant; dropout was tested at a single (too high) rate and failed; the best searched hyper-parameters were not carried over to the final baseline so that the three models stay comparable.
5. **Unseen objects are missed**: the transfer model detects only the tips of the hedgehog.
6. **Resolution** 640×360 blurs contours; in the notebook comparison figures the per-panel scores overlap the top image's lower edge slightly (cosmetic).
7. **Not comparable to the published benchmark** because we train on part of a test benchmark.
8. Experiments `20261104T172242…` and `20261104T182959…` and the four timestamped folders at the root are earlier runs kept for history; the **final experiment is `20261107T094734_134561Z`**.
