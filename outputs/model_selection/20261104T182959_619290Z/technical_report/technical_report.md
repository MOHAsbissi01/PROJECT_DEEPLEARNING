# Road Anomaly Segmentation on the RoadAnomaly Dataset

**Authors:** [Group member names]
**Module:** Deep Learning
**Experiment:** `20261104T182959_619290Z`

## 1. Problem statement

Autonomous vehicles must detect rare, unexpected obstacles (animals, rocks, cones, lost cargo)
that are absent from standard driving datasets. Given an RGB road image, we predict a per-pixel
anomaly score and a binary anomaly mask. The task is framed as **binary semantic segmentation**
(anomaly vs. background), complemented by a **reconstruction-based** approach inspired by
Lis et al. (ICCV 2019), who detect anomalies by resynthesising the image and comparing it with the input.

## 2. Dataset

* 60 RGB images (WebP), native size 1280x720, each with a semantic mask
  (`labels_semantic.png`, 0 = background, 2 = anomaly).
* Integrity checks: 60 valid pairs, 0 errors, every image contains an anomaly.
* Anomaly pixels: **9.85 %** of all pixels (strong class imbalance:
  a model predicting only background reaches 90.15 % accuracy,
  which is why accuracy is not reported as a main metric).
* Per-image anomaly area: median 6.62 %, min 0.37 %, max 42.33 %.

## 3. Methodology

### 3.1 Grouped split (leakage prevention)

Near-duplicate images (same scene or series) were grouped manually and each group was assigned to a
single partition (seed 42, 70 / 15 / 15 % of the groups).

| Partition | Images | Families |
| --- | --- | --- |
| train | 39 | animals: 22, cones: 6, obstacles: 9, vehicle: 2 |
| validation | 9 | animals: 3, cones: 1, lost_tires: 3, obstacles: 1, vehicle: 1 |
| test | 12 | animals: 6, cones: 1, obstacles: 5 |

Train fits the weights, validation selects checkpoints and decision thresholds, test is used **once** for the final evaluation.

### 3.2 Preprocessing

Images are resized to 640 x 360 (bilinear) and scaled to [0, 1]; masks are resized with nearest neighbour
and mapped 0/2 -> 0/1. Predictions are upsampled back to native resolution (bilinear) and evaluated
against the **original** masks.

### 3.3 Models

| Model | Description |
| --- | --- |
| baseline | Small encoder-decoder CNN trained from scratch (supervised segmentation). |
| transfer | U-Net with an ImageNet-pretrained MobileNetV2 encoder (transfer learning). |
| autoencoder | Convolutional autoencoder trained to reconstruct road background only; the per-pixel reconstruction error is the anomaly score (resynthesis idea, Lis et al. 2019). |

| Model | Trainable params | Frozen params | Loss | LR | Epochs | Best epoch | Train time (min) | Threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 46689 | 0 | bce_dice | 0.001 | 34 | 24 | 2.3 | 0.25 |
| transfer | 2690801 | 643520 | bce_dice | 0.0003 | 15 | 37 | 1.8 | 0.40 |
| autoencoder | 583499 | 3 | masked_mse_background | 0.001 | 60 | 50 | 11.2 | 0.05 |

All models use Adam, early stopping on validation loss and keep the best checkpoint. Thresholds are chosen on
validation by maximising the pooled pixel F1 (grid 0.05-0.95).

## 4. Results

### 4.1 Pooled pixel metrics

| Model | Set | Precision | Recall | F1 | Dice | IoU | AUROC | AP | FPR@95TPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | validation | 0.3364 | 0.6332 | 0.4393 | 0.4393 | 0.2815 | 0.8638 | 0.3903 | 0.4984 |
| baseline | test | 0.3415 | 0.6979 | 0.4586 | 0.4586 | 0.2975 | 0.8906 | 0.4207 | 0.3035 |
| transfer | validation | 0.9058 | 0.8736 | 0.8894 | 0.8894 | 0.8008 | 0.9947 | 0.9565 | 0.0280 |
| transfer | test | 0.8214 | 0.6816 | 0.7450 | 0.7450 | 0.5936 | 0.9452 | 0.8081 | 0.3119 |
| autoencoder | validation | 0.1022 | 0.6960 | 0.1782 | 0.1782 | 0.0978 | 0.5856 | 0.1014 | 0.8396 |
| autoencoder | test | 0.1215 | 0.8026 | 0.2110 | 0.2110 | 0.1179 | 0.6700 | 0.2487 | 0.8156 |

F1 and Dice are identical for binary masks. Validation scores were used for selection and are optimistic.
Primary model selected on validation: **transfer**. Best model on test (F1): **transfer**.

### 4.2 Per-image F1 on test

| Model | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- |
| baseline | 0.4506 | 0.4745 | 0.1106 | 0.8145 |
| transfer | 0.6879 | 0.7132 | 0.1827 | 0.9539 |
| autoencoder | 0.2053 | 0.1539 | 0.0337 | 0.5110 |

### 4.3 Test F1 / IoU per object family

| Model | Family | Images | F1 | IoU |
| --- | --- | --- | --- | --- |
| baseline | animals | 6 | 0.4952 | 0.3290 |
| baseline | cones | 1 | 0.2096 | 0.1171 |
| baseline | obstacles | 5 | 0.4198 | 0.2657 |
| transfer | animals | 6 | 0.8962 | 0.8120 |
| transfer | cones | 1 | 0.5212 | 0.3525 |
| transfer | obstacles | 5 | 0.4133 | 0.2605 |
| autoencoder | animals | 6 | 0.2693 | 0.1556 |
| autoencoder | cones | 1 | 0.0509 | 0.0261 |
| autoencoder | obstacles | 5 | 0.1588 | 0.0863 |

## 5. Visual results

**Figure 1.** `validation_threshold_curves`

![Figure 1](figures/01_validation_threshold_curves.png)

**Figure 2.** `test_metrics_comparison`

![Figure 2](figures/02_test_metrics_comparison.png)

**Figure 3.** `roc_pr_curves`

![Figure 3](figures/03_roc_pr_curves.png)

**Figure 4.** `animals12_Collared_Peccary_crossing_the_road_comparison`

![Figure 4](figures/04_animals12_Collared_Peccary_crossing_the_road_comparison.png)

**Figure 5.** `autoencoder_reconstructions`

![Figure 5](figures/05_autoencoder_reconstructions.png)

**Figure 6.** `obstacles06_cardboard_czech_hedgehog_comparison`

![Figure 6](figures/06_obstacles06_cardboard_czech_hedgehog_comparison.png)

**Figure 7.** `obstacles07_czech_hedgehog_comparison`

![Figure 7](figures/07_obstacles07_czech_hedgehog_comparison.png)

**Figure 8.** `per_image_f1`

![Figure 8](figures/08_per_image_f1.png)


## 6. Critical analysis

[TO COMPLETE after reviewing the figures: compare precision/recall trade-offs, typical false alarms
(road markings, shadows, vehicles), missed anomalies (small or road-coloured objects), contour quality,
and the validation-test gap (small sets: 9 and 12 images, so one difficult scene changes the scores).]

## 7. Limitations and future work

* Only 60 images: high variance of scores and thresholds; the original authors designed RoadAnomaly as a
  *test-only* benchmark, so supervised training on part of it is a pedagogical protocol, not the original benchmark.
* Pixels of the same image are not independent; per-image metrics are reported to show the spread.
* Possible improvements: pre-training on a large driving dataset (Cityscapes) and anomaly data synthesis,
  stronger augmentation, encoder fine-tuning, Dice/focal losses, higher input resolution for small objects,
  and the full resynthesis pipeline of Lis et al. (semantic segmentation -> GAN resynthesis -> discrepancy network).

## 8. Conclusion

[TO COMPLETE: answer the objective in 4-5 sentences using the numbers above.]

## References

* K. Lis, K. Nakka, P. Fua, M. Salzmann. *Detecting the Unexpected via Image Resynthesis.* ICCV 2019.
* M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
* O. Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
* RoadAnomaly dataset documentation and `credits.txt` (image sources).

Numerical evidence: see `evidence/`.
