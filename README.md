# RARE26 Challenge: Barrett's Neoplasia Detection

**Barrett's neoplasia detection for the RARE26 challenge with a k-fold ensemble of DINOv2 ViT-B/14 models.**

This project contains the training, inference, and synthetic-data code behind our submission to the
**RARE26 challenge** (hosted on Grand-Challenge). The task is binary classification of
endoscopy images of Barrett's esophagus: **neoplasia (`neo`)** vs. **non-dysplastic Barrett's esophagus (`ndbe`)**.
Positives are rare (about 6% of the data), so the challenge scores models on precision at high recall,
reported here as **PPV@90% Recall**, together with AUPRC.

The final submission is an ensemble of **3 DINOv2 ViT-B/14 configurations**, giving
**13 fold checkpoints** in total. Their predicted probabilities are averaged.

---

## Table of contents

1. [Final models](#1-final-models)
2. [Cross-validation results](#2-cross-validation-results)
3. [Method](#3-method)
4. [Data](#4-data)
5. [Project layout](#5-project-layout)
6. [Installation](#6-installation)
7. [Reproducing training](#7-reproducing-training)
8. [Inference](#8-inference)
9. [Synthetic data generation](#9-synthetic-data-generation)
10. [Decoding the run names](#10-decoding-the-run-names)
11. [Notes and limitations](#11-notes-and-limitations)
12. [Acknowledgements and references](#12-acknowledgements-and-references)

---

## 1. Final models

All three models share the same backbone and recipe: a DINOv2 ViT-B/14 at 336×336 with the patch embedding and
first 4 transformer blocks frozen, a single linear head, and model selection on `PPV + PPV@90% Recall`.
They differ in **which extra datasets they train on** and **how the folds are split**.

| # | Short name | Folds | Batch | Extra training data (besides center_1, center_2, EDD2020) | Weights folder in `trained_models/` |
|---|---|---|---|---|---|
| **A** | `dinov2-B14 / multi-source / 5-fold` | 5 | 32 | GastroVision, **synthetic (UniMedVL)**, Barrett archive | `timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs32_no_endo_edd2020_with_ndbe_train_no_hypershortseg_gastrovision_syntheticdata_barett_archive_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_RemoveShiftInLightEDD` |
| **B** | `dinov2-B14 / red-patch / 4-fold / v0.02` | 4 | 16 | red_patch | `timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs16_no_endo_edd2020_with_ndbe_train_no_hypershortseg_no_gastrovision_no_syntheticdata_no_barett_archive_redpatch_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_v0.02_4fold_` |
| **C** | `dinov2-B14 / red-patch / 4-fold / v0.01` | 4 | 16 | red_patch | `timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs16_no_endo_edd2020_with_ndbe_train_no_hypershortseg_no_gastrovision_no_syntheticdata_no_barett_archive_redpatch_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_v0.01_4fold_` |

Each weights folder contains:

```
best_model_fold{0..K-1}.pth   # state_dict of BinaryClassifier (backbone + linear head), ~344 MB each
cv_results.csv                # per-fold best epoch + validation metrics, plus the cross-fold means
folds_copy.json               # exact filenames/labels and fold assignment used for this run
```

Models **B** and **C** use **identical fold splits**: their `folds_copy.json` files are byte-for-byte equal.
They are two independent training runs of the same configuration, tagged `v0.01` and `v0.02`.
Training is not bit-wise deterministic on GPU, so the two runs converge to different checkpoints
and different best epochs. Combining them works like seed bagging and reduces variance.

---

## 2. Cross-validation results

The table shows the mean of the per-fold best checkpoints on each fold's held-out validation split, taken from
`cv_results.csv`. Accuracy, PPV, sensitivity, and specificity use a threshold of 0.5.
PPV@90% Recall is the precision at the operating point on the precision-recall curve
where recall is at least 0.9.

| Model | AUROC | AUPRC | **PPV@90% Recall** | PPV | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|---|
| **A** (5-fold, multi-source + synthetic) | 0.9839 | **0.9439** | 0.8566 | **0.9268** | 0.8809 | 0.9954 | 0.9882 |
| **B** (4-fold, red_patch, v0.02) | 0.9878 | 0.9395 | 0.8601 | 0.9222 | **0.8835** | 0.9950 | **0.9881** |
| **C** (4-fold, red_patch, v0.01) | **0.9883** | 0.9377 | **0.8969** | 0.9211 | 0.8734 | 0.9950 | 0.9874 |

<details>
<summary>Per-fold breakdown</summary>

**Model A**: 5 folds. Each fold has about 2,574 training images (about 160 positive) and about 644 validation images (about 40 positive).

| Fold | Best epoch | AUROC | AUPRC | PPV@90%R | PPV | Sens | Spec |
|---|---|---|---|---|---|---|---|
| 0 | 15 | 0.9991 | 0.9868 | 0.9737 | 0.9512 | 0.9512 | 0.9967 |
| 1 | 11 | 0.9685 | 0.9181 | 0.8085 | 0.9189 | 0.8095 | 0.9950 |
| 2 | 8  | 0.9879 | 0.9433 | 0.9231 | 0.9000 | 0.9000 | 0.9934 |
| 3 | 15 | 0.9816 | 0.9370 | 0.9231 | 0.9211 | 0.8974 | 0.9950 |
| 4 | 11 | 0.9824 | 0.9345 | 0.6545 | 0.9429 | 0.8462 | 0.9967 |

**Model B (v0.02)**: 4 folds. Each fold has about 2,386 training images (about 148 positive) and about 796 validation images (about 49 positive).

| Fold | Best epoch | AUROC | AUPRC | PPV@90%R | PPV | Sens | Spec |
|---|---|---|---|---|---|---|---|
| 0 | 2  | 0.9825 | 0.9361 | 0.9787 | 0.9783 | 0.8824 | 0.9987 |
| 1 | 17 | 0.9949 | 0.9531 | 0.8824 | 0.8824 | 0.9184 | 0.9920 |
| 2 | 14 | 0.9922 | 0.9495 | 0.9362 | 0.8980 | 0.9167 | 0.9933 |
| 3 | 12 | 0.9816 | 0.9191 | 0.6429 | 0.9302 | 0.8163 | 0.9960 |

**Model C (v0.01)**: same folds as B.

| Fold | Best epoch | AUROC | AUPRC | PPV@90%R | PPV | Sens | Spec |
|---|---|---|---|---|---|---|---|
| 0 | 2  | 0.9761 | 0.9166 | 0.9583 | 0.9778 | 0.8627 | 0.9987 |
| 1 | 18 | 0.9958 | 0.9515 | 0.8333 | 0.8776 | 0.8776 | 0.9920 |
| 2 | 18 | 0.9945 | 0.9583 | 0.9778 | 0.9565 | 0.9167 | 0.9973 |
| 3 | 18 | 0.9868 | 0.9245 | 0.8182 | 0.8723 | 0.8367 | 0.9920 |

</details>

> **Caveat.** Model A's numbers are **not directly comparable** with B and C. It uses a different
> number of folds and a different validation pool: its validation folds also contain GastroVision, synthetic, and
> Barrett-archive images, while B and C validate on center_1, center_2, EDD2020, and red_patch.
> PPV@90% Recall also varies a lot from fold to fold, because each fold has only 40 to 50 positives.
> This variance is the main reason for ensembling.

---

## 3. Method

### 3.1 Architecture

```
image (RGB) ──► Resize 336×336 (bicubic, antialias) ──► Normalize (ImageNet mean/std)
           ──► DINOv2 ViT-B/14 backbone (timm `vit_base_patch14_dinov2.lvd142m`, img_size=336, num_classes=0)
           ──► CLS embedding (768-d)
           ──► nn.Linear(768, 1) ──► logit ──► sigmoid ──► P(neoplasia)
```

* **Backbone initialisation:** `pretrained_weights/dinov2.pth`, a DINOv2 training checkpoint.
  The `teacher` state dict is loaded with the `backbone.` prefix stripped. The DINO head, the mask token, and the
  register tokens are dropped, so they show up as unexpected keys.
* **Partial freezing** (`--freeze_layers 4`): the patch embedding, the CLS token, and transformer blocks 0–3 are frozen.
  Blocks 4–11, the final norm, and the head are fine-tuned.
* **Head:** a single linear layer that outputs one logit (`BinaryClassifier` in `finetune_code/train_ensemble.py`).

### 3.2 Training recipe

| Setting | Value |
|---|---|
| Optimiser | AdamW, lr `2e-5`, weight decay `1e-2` |
| LR schedule | Linear warm-up for the first 10% of steps (start factor 0.1), then cosine annealing |
| Epochs | 20 per fold |
| Batch size | 32 (A) / 16 (B, C) |
| Loss | `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos`, computed per fold |
| Sampling | Standard shuffle, no over- or under-sampling |
| Input size | 336×336, bicubic interpolation with antialiasing |
| CV | Stratified K-fold, seed 42, stratified by source group and label. Fold membership is cached in `data_for_modeling/master_folds_*.json` so runs can be reproduced |
| Hardware | 2× NVIDIA RTX 6000 on an LSF cluster (`bsub`) |

### 3.3 Data augmentation (training only)

The order matches `get_train_transforms` in `finetune_code/utils.py`:

1. **Resize** to 336×336 (bicubic, antialias).
2. **Zoom / blur / distortion** (`--use_zoom_blur_distortion yes`). With p = 0.5, apply **exactly one** of:
   * zoom-in: `A.Affine(scale=1.0–1.08)`
   * blur or noise: one of Motion, Median, or Gaussian blur (kernel 3), or GaussNoise with variance 3–18
   * geometric distortion: one of Optical, Grid, or Elastic distortion
3. `ToTensor`.
4. **Random black boxes** (`--use_blackbox yes`, the "blackboxv2" tag). With p = 0.4, draw 5–10 black 4×10 px rectangles.
   They mimic specular-highlight removal and small occlusions.
5. **Random rotation** between 15° and 355°.
6. **Random vertical flip** with p = 0.35.
7. **Colour jitter** with p = 0.7: brightness 0.8–1.2, contrast 0.9–1.1, saturation 0.95–1.05, hue ±0.01.
   The ranges are kept narrow on purpose to preserve mucosal colour and vascular-pattern cues.
8. Normalize with the ImageNet mean and std.

Validation and inference only resize, convert to a tensor, and normalize.

### 3.4 Checkpoint selection

After each epoch, the model is evaluated on the fold's validation split. The checkpoint with the highest
**`PPV + PPV@90% Recall`** (`--best_model_metric "PPV+PPV@90% Recall"`) is kept. This rewards a model that is
precise at the default 0.5 threshold *and* at the high-recall operating point that matters clinically.
Metrics are rounded to 4 decimals before comparison. Ties are broken by the higher PPV (`--tiebreak_ppv`).

### 3.5 Ensembling

At inference, each of the 13 fold checkpoints (5 from A, 4 from B, 4 from C) outputs a sigmoid
probability for every image, and the final score is the **unweighted mean of all 13 probabilities**.
Because of this flat average, model A carries 5/13 of the weight and B and C together carry 8/13.

---

## 4. Data

All sources follow the same layout: `<source>/{neo,ndbe}/*.{png,jpg,jpeg}`. Labels come from the folder name.

| Source folder (`data/RARE26_train_data/`) | neo | ndbe | Used by |
|---|---:|---:|---|
| `center_1` (RARE26 official) | 61 | 2218 | A, B, C |
| `center_2` (RARE26 official) | 97 | 719 | A, B, C |
| `external_edd2020` (EDD2020) | 28 | 43 | A, B, C (included in the k-fold split) |
| `GastroVision` (Barrett's subset) | 8 | 32 | A |
| `synthetic_data` (UniMedVL-generated, see [§9](#9-synthetic-data-generation)) | 3 | 0 | A |
| `barett_archive` | 2 | 5 | A |
| `red_patch` (Barrett's images with red-patch findings) | 9 | 34 | B, C |
| `hyper_short_segment` | 1 | 11 | none (ablated) |
| `external_data_EVCBarrett` (Endovis) | 35 | 50 | none (ablated) |

The counts reflect the current state of the data directory. The training logs report 201 neo / 3,017 ndbe loaded
for model A and 197 neo / 3,017 ndbe for B and C. Some duplicate files were removed later, which
`--skip_missing_files` tolerates when it re-reads the cached folds.

> The data is **not** included in this project. Get the RARE26 data from the challenge page and the
> external datasets from their original sources, then arrange them in the layout above.

---

## 5. Project layout

```
RARE26_challenge/
├── finetune_code/
│   ├── train_ensemble.py   # main K-fold training script (all models in this README)
│   ├── dataset.py          # RareDataset: walks <source>/{neo,ndbe}/ folders
│   ├── utils.py            # train/val transforms and custom augmentations
│   ├── metrics.py          # AUROC, AUPRC, PPV@90% Recall, PPV, Sens, Spec, combined criteria
│   ├── infer_vitb.py       # offline inference on a flat folder of images -> CSV
│   └── test_models.py      # evaluate a trained ensemble on a labelled folder
├── aws_infer/
│   ├── inference.py        # Grand-Challenge container entry point (ensemble averaging)
│   └── model/timm_model.py # model builders + batched predict for the container
├── syn_generate/
│   └── text_to_image_unimedvl.py  # synthetic Barrett's / red-patch image generation
├── data_for_modeling/
│   ├── master_folds_*.json        # cached, reproducible fold assignments
│   └── oversample_list.txt        # optional per-file oversampling list (not used by A/B/C)
├── pretrained_weights/
│   └── dinov2.pth                 # DINOv2 ViT-B/14 teacher checkpoint (backbone init)
└── trained_models/
    ├── <model A folder>/best_model_fold{0..4}.pth, cv_results.csv, folds_copy.json
    ├── <model B folder>/best_model_fold{0..3}.pth, ...
    └── <model C folder>/best_model_fold{0..3}.pth, ...
```

> **Weights:** the 13 checkpoints are about 344 MB each, roughly 4.5 GB in total. That is too large to bundle with the code.
> Store them separately (e.g. shared storage or the Hugging Face Hub), then place them under `trained_models/`
> with the folder names exactly as listed in [§1](#1-final-models).

---

## 6. Installation

Tested environment (`rare26_vision_train` conda env):

| Package | Version |
|---|---|
| Python | 3.11 |
| torch / torchvision | 2.11.0 / 0.26.0 (CUDA 13.0) |
| timm | 1.0.28 |
| albumentations | 2.0.8 |
| scikit-learn | 1.9.0 |
| numpy | 2.4.4 |
| Pillow | 12.2 |

```bash
conda create -n rare26_vision_train python=3.11 -y
conda activate rare26_vision_train
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install timm==1.0.28 albumentations==2.0.8 scikit-learn pandas numpy pillow tqdm wandb
# optional, only for LoRA/DoRA GutCore experiments
pip install peft
```

> **Paths:** a few paths are hard-coded to the original cluster: the `trained_models/` output directory,
> `data_for_modeling/` for fold caches, and the default `--data_path_test` in `train_ensemble.py`.
> Search for `/home/chandraharsha.rachabathuni-umw` and point these at your own checkout before you run anything.
> Pretrained backbone weights are resolved as `<repo>/pretrained_weights/<--model_weights>`.

---

## 7. Reproducing training

All runs use `finetune_code/train_ensemble.py`. The run directory under `trained_models/` is named
automatically from the arguments (see [§10](#10-decoding-the-run-names)).

### Model A: 5-fold, multi-source + synthetic (exact command, from the job log)

```bash
python finetune_code/train_ensemble.py \
  --data_path data/RARE26_train_data \
  --batch_size 32 \
  --epochs 20 \
  --model_name "timm_dinov2_vitb_patch14" \
  --model_weights "dinov2.pth" \
  --backbone_dim 768 \
  --resize_img_dim 336 \
  --lr 2e-5 \
  --weight_decay 1e-2 \
  --antialias \
  --interpolation "bicubic" \
  --skip_missing_files \
  --tiebreak_ppv \
  --use_blackbox "yes" \
  --use_zoom_blur_distortion "yes" \
  --freeze_layers 4 \
  --use_extradata_synthetic_data \
  --use_extradata_barett_archive \
  --use_extradata_GastroVision \
  --best_model_metric "PPV+PPV@90% Recall" \
  --n_splits 5 \
  --use_wandb
```

### Models B and C: 4-fold, red_patch

These commands are reconstructed from the run-folder names. The two runs differ only in `--run_tag`,
and they share the same fold file (`folds_copy.json` is identical).

```bash
for TAG in v0.01_4fold v0.02_4fold; do
python finetune_code/train_ensemble.py \
  --data_path data/RARE26_train_data \
  --batch_size 16 \
  --epochs 20 \
  --model_name "timm_dinov2_vitb_patch14" \
  --model_weights "dinov2.pth" \
  --backbone_dim 768 \
  --resize_img_dim 336 \
  --lr 2e-5 \
  --weight_decay 1e-2 \
  --antialias \
  --interpolation "bicubic" \
  --skip_missing_files \
  --tiebreak_ppv \
  --use_blackbox "yes" \
  --use_zoom_blur_distortion "yes" \
  --freeze_layers 4 \
  --use_extradata_red_patch \
  --best_model_metric "PPV+PPV@90% Recall" \
  --n_splits 4 \
  --run_tag "$TAG" \
  --use_wandb
done
```

To reuse the exact folds, keep the corresponding `data_for_modeling/master_folds_base_data_*.json` files
(or copy a run's `folds_copy.json` there). If a cached fold file exists, the script loads it
instead of creating a new split.

### Useful flags

| Flag | Purpose |
|---|---|
| `--use_extradata_<SRC>` | Add a source to the k-fold pool (it is split across train and val) |
| `--<src>_train_only` | Add the source to the **training** side of every fold only, never to validation |
| `--exclude_edd2020` / `--eval_after_training` | Drop EDD2020 entirely, or hold it out as an ensemble test set |
| `--split_centers_train_test --center_test_size 0.25` | Hold out a stratified center_1/center_2 test set |
| `--use_tta --tta_transforms none hflip vflip` | TTA for held-out ensemble evaluation |
| `--loss focal`, `--use_sam`, `--sampling oversample` | Alternatives that were explored and not used in the final models |

---

## 8. Inference

### 8.1 Offline: folder of images to CSV

`infer_vitb.py` loads every `best_model_fold*.pth` in a weights folder, averages their probabilities, and writes a CSV.
Run it once per model, then average the three CSVs. Weight each model by its number of folds (5/4/4) if you want
results identical to the 13-checkpoint mean.

```bash
cd finetune_code
for M in \
  timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs32_no_endo_edd2020_with_ndbe_train_no_hypershortseg_gastrovision_syntheticdata_barett_archive_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_RemoveShiftInLightEDD \
  timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs16_no_endo_edd2020_with_ndbe_train_no_hypershortseg_no_gastrovision_no_syntheticdata_no_barett_archive_redpatch_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_v0.02_4fold_ \
  timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs16_no_endo_edd2020_with_ndbe_train_no_hypershortseg_no_gastrovision_no_syntheticdata_no_barett_archive_redpatch_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_v0.01_4fold_
do
  python infer_vitb.py \
    --weights_dir ../trained_models/$M \
    --data_path /path/to/images \
    --model_name timm_dinov2_vitb_patch14 \
    --output_csv preds_${M: -12}.csv
done
```

### 8.2 Minimal Python example (the full 13-checkpoint ensemble)

```python
import glob, torch, timm, numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MODEL_DIRS = [
    "trained_models/timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs32_no_endo_edd2020_with_ndbe_train_no_hypershortseg_gastrovision_syntheticdata_barett_archive_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_RemoveShiftInLightEDD",
    "trained_models/timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs16_no_endo_edd2020_with_ndbe_train_no_hypershortseg_no_gastrovision_no_syntheticdata_no_barett_archive_redpatch_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_v0.02_4fold_",
    "trained_models/timm_dinov2_vitb_patch14_weights_dinov2_freeze4_336px_bs16_no_endo_edd2020_with_ndbe_train_no_hypershortseg_no_gastrovision_no_syntheticdata_no_barett_archive_redpatch_best_ppv_ppv90r_tiebreakppv_blackboxv2_zoomblurdistortion_v3_v0.01_4fold_",
]

class BinaryClassifier(torch.nn.Module):
    def __init__(self, backbone, embed_dim=768):
        super().__init__()
        self.backbone, self.head = backbone, torch.nn.Linear(embed_dim, 1)
    def forward(self, x):
        return self.head(self.backbone(x))

tf = transforms.Compose([
    transforms.Resize((336, 336), interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.stack([tf(Image.open(p).convert("RGB")) for p in ["img1.jpg", "img2.jpg"]]).to(device)

probs = []
for d in MODEL_DIRS:
    for ckpt in sorted(glob.glob(f"{d}/best_model_fold*.pth")):
        bb = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=336)
        m = BinaryClassifier(bb).to(device).eval()
        m.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        with torch.no_grad():
            probs.append(torch.sigmoid(m(x)).squeeze(-1).cpu().numpy())
p_neoplasia = np.mean(probs, axis=0)   # mean over all 13 checkpoints
print(p_neoplasia)
```

### 8.3 Grand-Challenge container (`aws_infer/`)

`aws_infer/inference.py` reads the stacked input image
(`/input/images/stacked-barretts-esophagus-endoscopy`), runs every checkpoint in each folder listed in
`ensemble_dirs` under `resources/`, and writes the mean likelihoods to
`/output/stacked-neoplastic-lesion-likelihoods.json`.
To build the submission described here, set `ensemble_dirs` to the three folder names from [§1](#1-final-models)
and copy those folders into `resources/`. The `vitb_patch14` model type is selected automatically from the
`timm_dinov2_vitb_patch14` prefix.

---

## 9. Synthetic data generation

Model A adds a small set of **text-to-image synthetic neoplasia samples** generated with
[UniMedVL](https://github.com/uni-medical/UniMedVL), a unified medical vision-language model built on Bagel.
The script is `syn_generate/text_to_image_unimedvl.py`.

* Example prompt: *"Generate a good resolution endoscopy image with barrett esophagus and red patch darker than background on barrett esophagus section."*
* Settings: 336×336 output, `cfg_text_scale=12.0`, `cfg_img_scale=1.0`, 500 timesteps, "think" mode on, seeds 32 and up.
* The selected generated images are stored as positives under `data/RARE26_train_data/synthetic_data/neo/` (3 images in the final training set).

```bash
conda env create -f syn_generate/environment.yaml   # -> rare26_env_syn
conda activate rare26_env_syn
# clone UniMedVL into syn_generate/UniMedVL_repo and download the HF checkpoint
# (General-Medical-AI/UniMedVL) into syn_generate/unimedvl_checkpoint
python syn_generate/text_to_image_unimedvl.py
```

---

## 10. Decoding the run names

Run folders are built by `build_modelname_suffix()` in `train_ensemble.py`:

| Token | Meaning |
|---|---|
| `timm_dinov2_vitb_patch14` | Backbone (`--model_name`) |
| `weights_dinov2` | Initialised from `pretrained_weights/dinov2.pth` |
| `freeze4` | Patch embedding + first 4 blocks frozen |
| `336px`, `bs16` / `bs32` | Input resolution, batch size |
| `no_endo` | Endovis (`external_data_EVCBarrett`) not used |
| `edd2020_with_ndbe_train` | EDD2020 (neo + ndbe) included in the k-fold pool |
| `no_hypershortseg` | Hyper-short-segment data not used |
| `gastrovision` / `no_gastrovision` | GastroVision on/off |
| `syntheticdata` / `no_syntheticdata` | UniMedVL synthetic images on/off |
| `barett_archive` / `no_barett_archive` | Barrett archive on/off |
| `redpatch` | red_patch data on |
| `best_ppv_ppv90r` | Checkpoint selected on `PPV + PPV@90% Recall` |
| `tiebreakppv` | Ties broken by PPV |
| `blackboxv2` | Random black-box augmentation |
| `zoomblurdistortion` | One-of zoom/blur/distortion augmentation |
| `v3`, `RemoveShiftInLightEDD`, `v0.01_4fold`, `v0.02_4fold` | Custom experiment tags (`--run_tag` / `config.custom_tags`) |

---

## 11. Notes and limitations

* **Small positive class.** Each validation fold has only 39–51 neoplasia images, so a single misranked positive
  moves PPV@90% Recall by several points. Treat the per-fold numbers as noisy.
* **CV is not an external test.** EDD2020 and the extra sources are mixed into the k-fold pool, so the CV
  estimates above are in-distribution. Use `--split_centers_train_test` or `--eval_after_training` for a held-out
  estimate.
* **Reconstructed commands.** The training logs for models B and C were not kept. Their commands in
  [§7](#7-reproducing-training) were rebuilt from the deterministic run-name builder and match it token for token.
* **Not a medical device.** This is research code for a challenge and is not intended for clinical use.

---

## 12. Acknowledgements and references

* **RARE26 challenge** organisers and the contributing centres for the Barrett's endoscopy data.
* **DINOv2**: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
* **timm**: Ross Wightman, PyTorch Image Models.
* **UniMedVL**: <https://github.com/uni-medical/UniMedVL> (synthetic image generation).
* **EDD2020**: Ali et al., *Endoscopy Disease Detection Challenge 2020*.
* **GastroVision**: Jha et al., *GastroVision: A Multi-class Endoscopy Image Dataset*, 2023.
* **Albumentations**: Buslaev et al., 2020.
