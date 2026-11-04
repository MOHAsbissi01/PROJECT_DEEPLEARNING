"""Generate the technical report (Markdown) from the files saved in an experiment folder.

Everything is read back from disk (validation_selection.json, test_results.json,
run configurations, dataset/model summaries, figures), so the report can be
regenerated at any time without re-running the experiments.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import config
from .inference import load_experiment

METRICS = ("precision", "recall", "f1", "dice", "iou")


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _pct(value, digits=2):
    return "n/a" if value is None else f"{100 * value:.{digits}f} %"


def _table(headers, rows):
    clean = lambda v: str(v).replace("|", r"\|")  # noqa: E731
    lines = ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
    return "\n".join(lines)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_report(selection_path: Path, authors: str = "[Group members]") -> Path:
    exp = load_experiment(selection_path)
    sel_dir = exp.directory
    payload, test = exp.payload, exp.test_results
    dataset = _read(sel_dir / "dataset_summary.json") or {}
    model_info = _read(sel_dir / "model_summaries.json") or {}
    names = exp.model_names
    splits = exp.splits or {}

    report_dir = sel_dir / "technical_report"
    figures_dir = report_dir / "figures"
    evidence_dir = report_dir / "evidence"
    figures_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(exist_ok=True)

    # ---------------- tables ----------------
    split_rows = []
    for split in ("train", "validation", "test"):
        members = splits.get(split, [])
        families = {}
        for n in members:
            fam = next((p for p in config.FILENAME_FAMILIES if n.startswith(p)), "other")
            families[fam] = families.get(fam, 0) + 1
        split_rows.append([split, len(members), ", ".join(f"{k}: {v}" for k, v in sorted(families.items()))])

    model_rows = []
    for name in names:
        info = model_info.get(name, {})
        cfg = exp.run_configs.get(name, {})
        summary = _read(sel_dir / f"{name}_training_summary.json") or {}
        model_rows.append([
            name,
            info.get("trainable", "n/a"),
            info.get("non_trainable", "n/a"),
            cfg.get("loss", "n/a"),
            cfg.get("learning_rate", "n/a"),
            summary.get("epochs_completed", "n/a"),
            summary.get("best_epoch", "n/a"),
            _fmt(summary.get("training_duration_seconds", None) and summary["training_duration_seconds"] / 60, 1),
            _fmt(exp.threshold(name), 2),
        ])

    result_rows = []
    for name in names:
        phases = [("validation", payload["models"][name]["validation_metrics"], payload["models"][name].get("ranking_metrics"))]
        if test and name in test["models"]:
            phases.append(("test", test["models"][name]["global_metrics"], test["models"][name].get("ranking_metrics")))
        for phase, scores, ranking in phases:
            ranking = ranking or {}
            result_rows.append([
                name, phase, *[_fmt(scores[m]) for m in METRICS],
                _fmt(ranking.get("auroc")), _fmt(ranking.get("average_precision")), _fmt(ranking.get("fpr_at_95_tpr")),
            ])

    per_image_rows, family_rows = [], []
    if test:
        for name in names:
            if name not in test["models"]:
                continue
            rows = test["models"][name]["per_image_metrics"]
            f1 = np.array([r["f1"] for r in rows])
            per_image_rows.append([name, _fmt(f1.mean()), _fmt(float(np.median(f1))), _fmt(f1.min()), _fmt(f1.max())])
            for fam, scores in sorted((test["models"][name].get("family_metrics") or {}).items()):
                family_rows.append([name, fam, scores.get("images", ""), _fmt(scores["f1"]), _fmt(scores["iou"])])

    # ---------------- figures ----------------
    figure_sources = [sel_dir / "validation_threshold_curves.png"]
    eval_dir = sel_dir / "final_evaluation"
    figure_sources += [eval_dir / "test_metrics_comparison.png", eval_dir / "roc_pr_curves.png"]
    if (eval_dir / "analysis").is_dir():
        figure_sources += sorted((eval_dir / "analysis").glob("*.png"))
    figure_md = []
    for i, src in enumerate([p for p in figure_sources if p.is_file()], start=1):
        dst = figures_dir / f"{i:02d}_{src.name}"
        shutil.copy2(src, dst)
        figure_md.append(f"**Figure {i}.** `{src.stem}`\n\n![Figure {i}](figures/{dst.name})\n")
    for src in [selection_path, eval_dir / "test_results.json", *sel_dir.glob("*_run_config.json")]:
        if Path(src).is_file():
            shutil.copy2(src, evidence_dir / Path(src).name)

    primary = exp.primary_model
    best_test = max(test["models"], key=lambda n: test["models"][n]["global_metrics"]["f1"]) if test else None
    h, w = exp.image_size

    text = f"""# Road Anomaly Segmentation on the RoadAnomaly Dataset

**Authors:** {authors}
**Module:** Deep Learning
**Experiment:** `{exp.run_id}`

## 1. Problem statement

Autonomous vehicles must detect rare, unexpected obstacles (animals, rocks, cones, lost cargo)
that are absent from standard driving datasets. Given an RGB road image, we predict a per-pixel
anomaly score and a binary anomaly mask. The task is framed as **binary semantic segmentation**
(anomaly vs. background), complemented by a **reconstruction-based** approach inspired by
Lis et al. (ICCV 2019), who detect anomalies by resynthesising the image and comparing it with the input.

## 2. Dataset

* {dataset.get('n_images', 'n/a')} RGB images (WebP), native size {dataset.get('native_size', 'n/a')}, each with a semantic mask
  (`labels_semantic.png`, 0 = background, 2 = anomaly).
* Integrity checks: {dataset.get('valid_pairs', 'n/a')} valid pairs, {dataset.get('errors', 'n/a')} errors, every image contains an anomaly.
* Anomaly pixels: **{_pct(dataset.get('global_anomaly_fraction'))}** of all pixels (strong class imbalance:
  a model predicting only background reaches {_pct(1 - dataset['global_anomaly_fraction']) if 'global_anomaly_fraction' in dataset else 'n/a'} accuracy,
  which is why accuracy is not reported as a main metric).
* Per-image anomaly area: median {_pct(dataset.get('anomaly_fraction_median'))}, min {_pct(dataset.get('anomaly_fraction_min'))}, max {_pct(dataset.get('anomaly_fraction_max'))}.

## 3. Methodology

### 3.1 Grouped split (leakage prevention)

Near-duplicate images (same scene or series) were grouped manually and each group was assigned to a
single partition (seed {config.SEED}, 70 / 15 / 15 % of the groups).

{_table(["Partition", "Images", "Families"], split_rows)}

Train fits the weights, validation selects checkpoints and decision thresholds, test is used **once** for the final evaluation.

### 3.2 Preprocessing

Images are resized to {w} x {h} (bilinear) and scaled to [0, 1]; masks are resized with nearest neighbour
and mapped 0/2 -> 0/1. Predictions are upsampled back to native resolution (bilinear) and evaluated
against the **original** masks.

### 3.3 Models

| Model | Description |
| --- | --- |
""" + "\n".join(f"| {n} | {config.MODEL_DESCRIPTIONS.get(n, '')} |" for n in names) + f"""

{_table(["Model", "Trainable params", "Frozen params", "Loss", "LR", "Epochs", "Best epoch", "Train time (min)", "Threshold"], model_rows)}

All models use Adam, early stopping on validation loss and keep the best checkpoint. Thresholds are chosen on
validation by maximising the pooled pixel F1 (grid 0.05-0.95).

## 4. Results

### 4.1 Pooled pixel metrics

{_table(["Model", "Set", "Precision", "Recall", "F1", "Dice", "IoU", "AUROC", "AP", "FPR@95TPR"], result_rows)}

F1 and Dice are identical for binary masks. Validation scores were used for selection and are optimistic.
Primary model selected on validation: **{primary}**.{f" Best model on test (F1): **{best_test}**." if best_test else ""}

### 4.2 Per-image F1 on test

{_table(["Model", "Mean", "Median", "Min", "Max"], per_image_rows) if per_image_rows else "_Test evaluation not run yet._"}

### 4.3 Test F1 / IoU per object family

{_table(["Model", "Family", "Images", "F1", "IoU"], family_rows) if family_rows else "_Not available._"}

## 5. Visual results

{chr(10).join(figure_md) if figure_md else "_No figures found._"}

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
"""
    path = report_dir / "technical_report.md"
    path.write_text(text, encoding="utf-8")
    return path
