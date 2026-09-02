import os
import sys
import csv
import copy
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import timm
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from huggingface_hub import hf_hub_download
from torchvision.models import resnet50

from dataset import RareDataset, RareTestSet
from utils import get_train_transforms, get_val_transforms
from metrics import compute_metrics


class config:
    train_progress_bar = False
    best_model_metric = "PPV@90% Recall"
    BASE_DIR = Path(__file__).resolve().parent.parent
    modelname_suffix = None  # set at runtime by build_modelname_suffix(args)
    custom_tags = ['']


def get_edd2020_usage(args):
    """Determine how the EDD2020 external dataset is used for this run.

    - "excluded": not added to train and not used as a held-out test set
      (ablation via --exclude_edd2020, to measure impact of this data being
      absent from the pipeline entirely)
    - "test": excluded from train and evaluated afterwards as an ensemble
      test set (--eval_after_training)
    - "train": added to the k-fold training data
    """
    if args.exclude_edd2020:
        return "excluded"
    return "test" if args.eval_after_training else "train"


VIT_BACKBONE_PREFIXES = (
    "torch_dinov3_vitl",
    "timm_dinov2_vitb_patch14",
    "torch_dinov3_vits16",
    "timm_dinov3_vitl",
    "vits16",
    "timm_gutcore_vitl_patch14",
)

RESNET_BACKBONE_PREFIXES = (
    "resnet50",
)




def is_vit_backbone(model_name):
    """Whether model_name builds a ViT-style backbone (patch_embed/cls_token/blocks attrs).

    All of torch_dinov3_vitl16/vitb16/vits16, timm_dinov3_vitl16, timm_dinov2_vitb_patch14,
    timm_gutcore_vitl_patch14, and vits16 (timm vit_small_patch16_224) expose this same
    attribute shape, which is what freeze_vit_layers() relies on -- so --freeze_layers is
    supported for any of them, regardless of --model_weights. resnet50 has an unrelated
    internal structure and is handled separately by is_resnet_backbone()/freeze_resnet_layers();
    maxvit_tiny, mambavision, and mamba have their own unrelated structures and are not
    supported.
    """
    return model_name.startswith(VIT_BACKBONE_PREFIXES)


GUTCORE_NORMALIZATION = {
    "mean": [0.570873, 0.328645, 0.251895],
    "std": [0.273905, 0.198141, 0.171382],
}


def get_normalization_stats(model_name):
    """Return the (mean, std) a backbone's own pretraining expects, or (None, None).

    None falls back to get_train_transforms/get_val_transforms' ImageNet default. GutCore-ViT-L
    was pretrained on endoscopy images normalized with its own per-channel stats (see
    github.com/SMC-GutX/GutCore/blob/main/src/gutcore/config.py) rather than ImageNet's --
    using ImageNet stats instead would feed it a systematically shifted input distribution,
    which its frozen patch_embed (see freeze_vit_layers) has no way to correct for.
    """
    if model_name.startswith("timm_gutcore_vitl_patch14"):
        return GUTCORE_NORMALIZATION["mean"], GUTCORE_NORMALIZATION["std"]
    return None, None


def is_resnet_backbone(model_name):
    """Whether model_name builds a resnet50-style backbone (conv1/bn1/layer1-4 attrs).

    Used by freeze_resnet_layers() to gate --freeze_layers support for resnet50,
    the same way is_vit_backbone() gates it for ViT-style backbones.
    """
    return model_name.startswith(RESNET_BACKBONE_PREFIXES)


BEST_MODEL_METRIC_TAGS = {
    "Loss": "loss",
    "AUROC": "auroc",
    "AUPRC": "auprc",
    "PPV@90% Recall": "ppv90r",
    "PPV": "ppv",
    "Accuracy": "acc",
    "Sensitivity": "sens",
    "Specificity": "spec",
    "PPV+PPV@90% Recall": "ppv_ppv90r",
    "PPV@90% Recall+AUPRC": "ppv90r_auprc",
}


def build_modelname_suffix(args):
    parts = [args.model_name]

    if args.model_weights:
        parts.append(f"weights_{Path(args.model_weights).stem}")
    elif args.backbone_pretrained_flag:
        parts.append("pretrained")
    else:
        parts.append("scratch")

    if (is_vit_backbone(args.model_name) or is_resnet_backbone(args.model_name)) and args.freeze_layers > 0:
        parts.append(f"freeze{args.freeze_layers}")

    if args.use_lora:
        parts.append(f"lora_r{args.lora_rank}_a{args.lora_alpha}")
    elif args.use_dora:
        parts.append(f"dora_r{args.lora_rank}_a{args.lora_alpha}")

    parts.append(f"{args.resize_img_dim}px")
    parts.append(f"bs{args.batch_size}")

    # scaling_tag = args.interpolation + ("_antialias" if args.antialias else "_noantialias")
    # parts.append(scaling_tag)

    if args.neo_only_other_sources:
        # Supersedes the endo/hss/gastrovision/barett_archive tags below -- all of
        # those sources are folded into this single "neo-only" tag instead.
        parts.append("neoonly_other_sources")
    else:
        parts.append("endo" if args.use_extradata_Endovis else "no_endo")

    edd2020_tag = f"edd2020_with_ndbe_{get_edd2020_usage(args)}"
    if get_edd2020_usage(args) == "train" and args.edd2020_train_only:
        edd2020_tag += "_trainonly"
    parts.append(edd2020_tag)

    if not args.neo_only_other_sources:
        hss_tag = "hypershortseg" if args.use_extradata_hyper_short_segment else "no_hypershortseg"
        if args.use_extradata_hyper_short_segment and args.hyper_short_segment_train_only:
            hss_tag += "_trainonly"
        parts.append(hss_tag)

        gastrovision_tag = "gastrovision" if args.use_extradata_GastroVision else "no_gastrovision"
        if args.use_extradata_GastroVision and args.gastrovision_train_only:
            gastrovision_tag += "_trainonly"
        parts.append(gastrovision_tag)

        synthetic_data_tag = "syntheticdata" if args.use_extradata_synthetic_data else "no_syntheticdata"
        if args.use_extradata_synthetic_data and args.synthetic_data_train_only:
            synthetic_data_tag += "_trainonly"
        parts.append(synthetic_data_tag)

        barett_archive_tag = "barett_archive" if args.use_extradata_barett_archive else "no_barett_archive"
        if args.use_extradata_barett_archive and args.barett_archive_train_only:
            barett_archive_tag += "_trainonly"
        parts.append(barett_archive_tag)

        red_patch_tag = "redpatch" if args.use_extradata_red_patch else "no_redpatch"
        if args.use_extradata_red_patch and args.red_patch_train_only:
            red_patch_tag += "_trainonly"
        parts.append(red_patch_tag)

    if args.loss == "focal":
        parts.append(f"focal_g{args.focal_gamma}_a{args.focal_alpha}")

    if args.use_sam:
        sam_tag = f"sam_rho{args.sam_rho}"
        if args.sam_adaptive:
            sam_tag += "_adaptive"
        parts.append(sam_tag)

    parts.append(f"best_{BEST_MODEL_METRIC_TAGS[args.best_model_metric]}")

    if args.tiebreak_ppv and args.best_model_metric != "PPV":
        parts.append("tiebreakppv")

    if args.split_centers_train_test:
        parts.append(f"centerholdout{args.center_test_size}")

    if args.use_tta:
        parts.append("tta_" + "-".join(args.tta_transforms))

    if args.use_blackbox:
        parts.append("blackboxv2")

    if args.use_blur_noise:
        parts.append("blurnoise")

    if args.use_distortion:
        parts.append("distortion")

    if args.use_zoom:
        parts.append("zoom5")

    if args.use_black_border:
        parts.append("blackborder90")

    if args.use_zoom_blur_distortion:
        parts.append("zoomblurdistortion")

    if args.center2_pos_upsample_factor > 1.0:
        parts.append(f"center2posup{args.center2_pos_upsample_factor}")

    if args.oversample_filelist and args.oversample_filelist_factor > 1.0:
        parts.append(f"flp{args.oversample_filelist_factor}")

    parts.append("")

    if args.run_tag:
        parts.append(args.run_tag)

    return "_".join(parts + config.custom_tags)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sample_label(sample_label):
    if isinstance(sample_label, str):
        if sample_label == "nondysplastic":
            return 0
        elif sample_label == "neoplasia":
            return 1
        else:
            raise ValueError(f"Unexpected label string: {sample_label}")
    return int(sample_label)


def get_labels_from_dataset(dataset):
    labels = []
    for _, label in dataset.samples:
        labels.append(get_sample_label(label))
    return np.array(labels)


def get_filenames_from_dataset(dataset):
    return [Path(img_path).name for img_path, _ in dataset.samples]


def is_endovis_sample(img_path):
    return "external_data_EVCBarrett" in Path(img_path).parts


def is_edd2020_sample(img_path):
    return "external_edd2020" in Path(img_path).parts


def is_hyper_short_segment_sample(img_path):
    return "hyper_short_segment" in Path(img_path).parts


def is_gastrovision_sample(img_path):
    return "GastroVision" in Path(img_path).parts


def is_red_patch_sample(img_path):
    return "red_patch" in Path(img_path).parts


def is_synthetic_data_sample(img_path):
    return "synthetic_data" in Path(img_path).parts


def is_barett_archive_sample(img_path):
    return "barett_archive" in Path(img_path).parts


def is_center_sample(img_path, center_names=("center_1", "center_2")):
    return any(name in Path(img_path).parts for name in center_names)


def is_other_source_sample(img_path):
    """Whether img_path belongs to any source folder other than center_1/center_2/edd2020.

    Matched by path, not by which --use_extradata_* flags are set, so it covers every
    such folder present on disk -- Endovis, hyper_short_segment, GastroVision,
    synthetic_data, barett_archive, red_patch, external_data_hyper, and any future addition --
    used by --neo_only_other_sources to restrict those folders to their 'neo' images
    only.
    """
    return not is_center_sample(img_path) and not is_edd2020_sample(img_path)


def upsample_group_in_train_indices(dataset, train_indices, factor, center_name="center_2",
                                     target_label=1, seed=42):
    """Duplicate {center_name}-{target_label} samples within train_indices by `factor`.

    factor <= 1.0 is a no-op. factor=2.0 duplicates every matching sample once (so
    it appears twice in the returned index list); non-integer factors duplicate the
    integer part for every matching sample and add one extra copy for a random
    fraction of them, so the expected upsample ratio matches `factor` exactly.
    Only affects the returned list -- val/test indices are untouched, and the
    underlying dataset/fold split is not modified.
    """
    if factor <= 1.0:
        return train_indices

    matching = [
        idx for idx in train_indices
        if is_center_sample(dataset.samples[idx][0], (center_name,))
        and get_sample_label(dataset.samples[idx][1]) == target_label
    ]
    if not matching:
        print(f"Upsampling requested for {center_name} class {target_label}, but no "
              f"matching samples found in this fold's training set -- skipping.")
        return train_indices

    n_repeat = int(factor)
    extra_fraction = factor - n_repeat

    duplicated = matching * (n_repeat - 1)
    if extra_fraction > 0:
        n_extra = int(round(extra_fraction * len(matching)))
        duplicated += random.Random(seed).sample(matching, min(n_extra, len(matching)))

    print(f"Upsampling {center_name} class {target_label} (factor={factor}): "
          f"{len(matching)} matching samples -> {len(duplicated)} duplicated copies added.")

    return train_indices + duplicated


def load_oversample_filelist(path):
    """Read a newline-delimited file naming samples to slightly oversample.

    Blank lines and lines starting with '#' are ignored. Each entry is reduced to
    its basename, so the file may list bare filenames or full paths -- matching
    only cares about the basename, same as resolve_indices_from_filenames.
    """
    with open(path, "r") as fp:
        lines = [line.strip() for line in fp]
    return [Path(line).name for line in lines if line and not line.startswith("#")]


def upsample_filenames_in_train_indices(dataset, train_indices, filenames, factor,
                                         seed=42, skip_missing=False):
    """Duplicate an explicit list of filenames within train_indices by `factor`.

    Mirrors upsample_group_in_train_indices, but matches an arbitrary hand-picked
    filename list (e.g. from --oversample_filelist) instead of a center_name/
    target_label group. factor <= 1.0 is a no-op. factor=2.0 duplicates every
    matching sample once (so it appears twice in the returned index list);
    non-integer factors duplicate the integer part for every matching sample and
    add one extra copy for a random fraction of them, so the expected upsample
    ratio matches `factor` exactly. Only affects the returned list -- val/test
    indices are untouched, and the underlying dataset/fold split is not modified.
    """
    if factor <= 1.0:
        return train_indices

    target_indices = set(
        resolve_indices_from_filenames(dataset, filenames, skip_missing=skip_missing)
    )
    matching = [idx for idx in train_indices if idx in target_indices]
    if not matching:
        print(f"Oversample filelist requested ({len(filenames)} filenames), but none of "
              f"them are present in this fold's training set -- skipping.")
        return train_indices

    n_repeat = int(factor)
    extra_fraction = factor - n_repeat

    duplicated = matching * (n_repeat - 1)
    if extra_fraction > 0:
        n_extra = int(round(extra_fraction * len(matching)))
        duplicated += random.Random(seed).sample(matching, min(n_extra, len(matching)))

    print(f"Oversampling filelist (factor={factor}): {len(matching)} matching samples in "
          f"this fold's train set -> {len(duplicated)} duplicated copies added.")

    return train_indices + duplicated


def get_stratify_group(img_path, label, center_names=("center_1", "center_2")):
    """Combine center identity with class label into one k-fold stratification key.

    Stratifying on label alone only balances overall pos/neg counts per fold; it
    says nothing about how those samples split between center_1 and center_2.
    Keying on "{center}_{label}" instead makes StratifiedKFold balance
    center_1-negative, center_1-positive, center_2-negative, and
    center_2-positive counts roughly evenly across every fold's train and val
    portions. Samples from any other source (Endovis/EDD2020/hyper external
    data) fall back to a single "other_{label}" group per label, matching prior
    label-only behavior for that data.
    """
    if is_center_sample(img_path, center_names):
        center = next(name for name in center_names if name in Path(img_path).parts)
        return f"{center}_{label}"
    return f"other_{label}"


def split_center_holdout(dataset, test_size=0.2, seed=42, center_names=("center_1", "center_2")):
    """Hold out a stratified test split of filenames from center_1/center_2 samples only.

    The split is computed purely from center_1/center_2 filenames and labels, sorted
    into a deterministic order. It deliberately never looks at dataset-wide sample
    positions, because those positions shift depending on which --use_extradata_*
    flags (Endovis, EDD2020) are enabled for a given run -- those flags only add or
    remove samples from other folders, never from center_1/center_2, but they do
    change where center_1/center_2 samples land in `dataset.samples`. Keying the
    split off filenames instead of indices guarantees the same physical images are
    held out regardless of which additional-data flags a run uses.

    Returns a sorted list of held-out filenames (not indices). Use
    `resolve_indices_from_filenames` to map these back to indices for a specific
    dataset instance.
    """
    center_samples = sorted(
        (
            (Path(img_path).name, get_sample_label(label))
            for img_path, label in dataset.samples
            if is_center_sample(img_path, center_names)
        ),
        key=lambda sample: sample[0],
    )
    n_extra = len(dataset.samples) - len(center_samples)

    if not center_samples:
        raise ValueError("No center_1/center_2 samples found to build a holdout split from.")

    center_filenames = [fn for fn, _ in center_samples]
    center_labels = [label for _, label in center_samples]

    if len(set(center_labels)) < 2:
        raise ValueError("Cannot stratify center train/test split: fewer than 2 classes present.")

    _, holdout_filenames = train_test_split(
        center_filenames,
        test_size=test_size,
        random_state=seed,
        stratify=center_labels,
    )
    holdout_filenames = sorted(holdout_filenames)

    print(
        f"Center train/test split: {len(center_filenames) - len(holdout_filenames)} of {len(center_filenames)} "
        f"center_1/center_2 samples kept for k-fold training, {len(holdout_filenames)} held out as test "
        f"(split is filename-based, independent of --use_extradata_* flags). "
        f"{n_extra} additional-data samples (Endovis/EDD2020, if enabled) are unaffected and remain "
        f"in the k-fold training pool."
    )
    return holdout_filenames


def resolve_indices_from_filenames(dataset, filenames, skip_missing=False):
    """Map a list of filenames to this dataset instance's current sample indices.

    Raw indices are not stable across dataset instances built with different
    --use_extradata_* flags, or after files are added/removed on disk (e.g.
    deleting duplicate Endovis/EVCBarrett images), so cached splits are stored
    as filenames and resolved to indices fresh against whichever dataset
    instance is in use.

    If skip_missing is False (default), a filename that no longer exists in the
    current dataset raises an error -- this is the safe default so silent data
    drift doesn't go unnoticed. If skip_missing is True, such filenames are
    dropped instead (use --skip_missing_files for this, e.g. after intentionally
    deleting duplicate files from a data folder).
    """
    fn_to_idx = {Path(img_path).name: i for i, (img_path, _) in enumerate(dataset.samples)}
    missing = [fn for fn in filenames if fn not in fn_to_idx]
    if missing and not skip_missing:
        raise ValueError(
            f"{len(missing)} filenames were not found in the current dataset "
            f"(e.g. {missing[:5]}). The data directory may have changed since this "
            f"split/fold metadata was created. Pass --skip_missing_files to drop "
            f"missing filenames instead of failing."
        )
    if missing:
        print(f"Skipping {len(missing)} filenames not found in the current dataset "
              f"(e.g. {missing[:5]}) -- likely deleted since this metadata was saved.")
    return [fn_to_idx[fn] for fn in filenames if fn in fn_to_idx]


def create_folds(dataset, n_splits=5, seed=42, exclude_indices=None):
    labels_full = get_labels_from_dataset(dataset)
    all_indices = np.arange(len(labels_full))

    if exclude_indices:
        exclude_mask = np.isin(all_indices, np.asarray(list(exclude_indices)))
        all_indices = all_indices[~exclude_mask]
        print(f"Excluding {int(exclude_mask.sum())} held-out samples (e.g. center test split) "
              f"from k-fold splitting.")

    labels = labels_full[all_indices]
    core_indices = all_indices
    core_labels = labels

    core_groups = np.array([
        get_stratify_group(dataset.samples[idx][0], label)
        for idx, label in zip(core_indices, core_labels)
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(core_indices, core_groups)):
        folds.append({
            "fold": fold_idx,
            "train_idx": core_indices[train_idx].tolist(),
            "val_idx": core_indices[val_idx].tolist(),
        })
    return folds


def save_folds_metadata(dataset, folds, out_path, holdout_indices=None):
    filenames = get_filenames_from_dataset(dataset)
    labels = get_labels_from_dataset(dataset).tolist()
    holdout_indices = holdout_indices or []

    payload = {
        "filenames": filenames,
        "labels": labels,
        "holdout_indices": holdout_indices,
        "holdout_filenames": [filenames[i] for i in holdout_indices],
        "holdout_labels": [labels[i] for i in holdout_indices],
        "folds": []
    }

    for f in folds:
        payload["folds"].append({
            "fold": f["fold"],
            "train_indices": f["train_idx"],
            "val_indices": f["val_idx"],
            "train_filenames": [filenames[i] for i in f["train_idx"]],
            "val_filenames": [filenames[i] for i in f["val_idx"]],
        })

    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Saved fold metadata to {out_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device,
                     grad_clip_mode=None, grad_clip_value=None, use_sam=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(dataloader, desc="Training", unit="batch",
              disable=not config.train_progress_bar) as pbar:
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            if use_sam:
                enable_running_stats(model)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                if grad_clip_mode == "global_norm":
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                elif grad_clip_mode == "value":
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_value)

                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                criterion(model(images), labels).backward()

                if grad_clip_mode == "global_norm":
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                elif grad_clip_mode == "value":
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_value)

                optimizer.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                if grad_clip_mode == "global_norm":
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                elif grad_clip_mode == "value":
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_value)

                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if config.train_progress_bar:
                pbar.set_postfix(
                    loss=total_loss / max(1, (pbar.n + 1)),
                    accuracy=correct / max(1, total)
                )

    return total_loss / len(dataloader), correct / total


TTA_TRANSFORMS = {
    "none": lambda x: x,
    "hflip": lambda x: torch.flip(x, dims=[-1]),
    "vflip": lambda x: torch.flip(x, dims=[-2]),
    "rot90": lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
    "rot180": lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
    "rot270": lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
}


def predict_with_tta(model, images, tta_transforms):
    """Average predicted probabilities over a batch across TTA views.

    Each entry in tta_transforms names a spatial transform (flip/rotation) applied
    directly to the already-resized/normalized image tensor -- these ops commute
    with per-channel normalization, so there's no need to un-normalize first.
    """
    probs_sum = None
    for name in tta_transforms:
        augmented = TTA_TRANSFORMS[name](images)
        probs = torch.sigmoid(model(augmented))
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(tta_transforms)


def round_metrics(metrics, ndigits=4):
    """Round every numeric value in a metrics dict to ndigits decimal places.

    NaN values (e.g. from a diverged fold) round to themselves. Rounding keeps
    metric comparisons (e.g. best-epoch selection, tiebreak_ppv) stable across
    floating-point noise that would otherwise make two epochs' scores differ in
    the 6th+ decimal place despite being effectively tied.
    """
    return {k: (round(v, ndigits) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().reshape(-1))
            all_scores.extend(probs.cpu().numpy().reshape(-1))

    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["Loss"] = total_loss / len(dataloader)
    metrics["Accuracy"] = correct / total
    return round_metrics(metrics)


def freeze_vit_layers(backbone, n_layers):
    """Freeze the patch/token embeddings and the first n_layers transformer blocks.

    Blocks beyond n_layers, the mask token, and the final norm stay trainable so
    the model can still adapt higher-level features during finetuning. Works for any
    ViT-style backbone matched by is_vit_backbone() -- they all share this attribute shape.
    """
    for param in backbone.patch_embed.parameters():
        param.requires_grad = False
    backbone.cls_token.requires_grad = False
    if hasattr(backbone, "storage_tokens"):
        backbone.storage_tokens.requires_grad = False

    n_layers = min(n_layers, len(backbone.blocks))
    for block in backbone.blocks[:n_layers]:
        for param in block.parameters():
            param.requires_grad = False

    print(f"Froze patch/token embeddings and first {n_layers} of {len(backbone.blocks)} "
          f"transformer blocks in the backbone.")


def freeze_resnet_layers(backbone, n_layers):
    """Freeze the stem (conv1/bn1) and the first n_layers residual stages.

    resnet50's four residual stages (layer1..layer4) are treated the same way
    freeze_vit_layers treats transformer blocks: the stem is always frozen, and
    n_layers counts how many of the four stages, in order, are frozen along with
    it. Stages beyond n_layers stay trainable so the model can still adapt
    higher-level features during finetuning. Works for any backbone matched by
    is_resnet_backbone() -- currently resnet50 only.
    """
    for param in backbone.conv1.parameters():
        param.requires_grad = False
    for param in backbone.bn1.parameters():
        param.requires_grad = False

    stages = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
    n_layers = min(n_layers, len(stages))
    for stage in stages[:n_layers]:
        for param in stage.parameters():
            param.requires_grad = False

    print(f"Froze stem (conv1/bn1) and first {n_layers} of {len(stages)} "
          f"residual stages in the resnet backbone.")


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter (Hu et al. 2021, LoRA).

    output = base(x) + scaling * lora_B(lora_A(dropout(x))), where scaling = alpha / rank.
    lora_B is zero-initialized so the adapter is a no-op at the start of finetuning --
    the wrapped layer's output exactly matches the frozen base checkpoint until training
    moves lora_B away from zero.
    """

    def __init__(self, base_linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False

        # dinov3's SelfAttention.compute_attention reads `self.qkv.in_features` directly
        # (not just through forward()), so this wrapper needs to expose the same nn.Linear-
        # like attributes the wrapped module would have.
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


def apply_lora_to_gutcore_backbone(backbone, rank=8, alpha=16, dropout=0.0,
                                    target_modules=("qkv", "proj", "w1", "w2", "w3")):
    """Freeze the whole GutCore-ViT-L backbone and inject trainable LoRA adapters (Hu et al.
    2021) into its attention (attn.qkv, attn.proj) and SwiGLU FFN (mlp.w1/w2/w3) Linear
    layers, in every transformer block.

    Only meaningful for the DINOv3-style backbone built in build_backbone() for
    timm_gutcore_vitl_patch14 -- it assumes each block exposes attn.{qkv,proj} and
    mlp.{w1,w2,w3} as plain nn.Linear submodules, which is that architecture's exact shape
    (confirmed against the released checkpoint's tensor keys). Not wired up for any other
    --model_name.
    """
    for param in backbone.parameters():
        param.requires_grad = False

    n_replaced = 0
    for block in backbone.blocks:
        for name in target_modules:
            parent = block.attn if name in ("qkv", "proj") else block.mlp
            linear = getattr(parent, name)
            if not isinstance(linear, nn.Linear):
                raise TypeError(
                    f"--lora_target_modules entry {name!r} does not name an nn.Linear "
                    f"submodule of block.attn/block.mlp (got {type(linear)})."
                )
            setattr(parent, name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
            n_replaced += 1

    print(f"Applied LoRA (rank={rank}, alpha={alpha}, dropout={dropout}) to {n_replaced} "
          f"Linear layers ({sorted(set(target_modules))}) across {len(backbone.blocks)} "
          f"blocks; every other backbone parameter is frozen.")


def apply_dora_to_gutcore_backbone(backbone, rank=8, alpha=16, dropout=0.0,
                                    target_modules=("qkv", "proj", "w1", "w2", "w3")):
    """Freeze the whole GutCore-ViT-L backbone and inject trainable DoRA adapters (Liu et al.
    2024, https://arxiv.org/abs/2402.09353, "Weight-Decomposed Low-Rank Adaptation") into its
    attention (attn.qkv, attn.proj) and SwiGLU FFN (mlp.w1/w2/w3) Linear layers, in every
    transformer block, using HuggingFace's peft library instead of the hand-rolled LoRALinear
    class used by --use_lora.

    DoRA decomposes each targeted weight into a magnitude vector and a direction matrix, then
    applies the LoRA low-rank update to the direction component only, with the magnitude
    trained separately -- this tends to track full finetuning more closely than plain LoRA at
    the same rank. peft's LoraConfig(target_modules=...) matches any Linear submodule of the
    backbone by its final dotted name component, which for this architecture resolves to
    exactly attn.{qkv,proj} and mlp.{w1,w2,w3} in every block -- the same layers --use_lora
    targets. get_peft_model() freezes every backbone parameter except the injected adapters
    (including the DoRA magnitude vectors) by default. Only meaningful for the DINOv3-style
    backbone built in build_backbone() for timm_gutcore_vitl_patch14; not wired up for any
    other --model_name.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "--use_dora requires the `peft` library (pip install peft>=0.9.0, the version "
            "that introduced LoraConfig(use_dora=True))."
        ) from exc

    dora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        use_dora=True,
        bias="none",
    )
    backbone = get_peft_model(backbone, dora_config)

    print(f"Applied DoRA (rank={rank}, alpha={alpha}, dropout={dropout}) via peft to Linear "
          f"layers named {sorted(set(target_modules))} across the backbone; every other "
          f"backbone parameter is frozen.")
    backbone.print_trainable_parameters()

    return backbone


def build_backbone(args, device):
    if args.model_name.startswith("torch_dinov3_vitl"):
        backbone = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vitl16",
            pretrained=args.backbone_pretrained_flag
        )
    elif args.model_name.startswith("timm_dinov2_vitb_patch14"):
        backbone = timm.create_model("timm/vit_base_patch14_dinov2.lvd142m",
                          pretrained=False,
                          num_classes=0,
                          img_size=336,
                          )

    elif args.model_name.startswith("torch_dinov3_vits16"):
        backbone = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vits16",
            pretrained=args.backbone_pretrained_flag
        )

    elif args.model_name.startswith("timm_dinov3_vitl"):
        backbone = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=args.backbone_pretrained_flag,
            num_classes=0
        )
    elif args.model_name.startswith("timm_gutcore_vitl_patch14"):
        # GutCore-ViT-L (https://github.com/SMC-GutX/GutCore) is a ViT-L/14 at 336px, but --
        # despite the "timm_" prefix kept here for backwards compatibility with existing run
        # configs -- it is NOT the plain timm dinov2 arch (learned pos_embed, standard MLP):
        # it's the facebookresearch/dinov3 DinoVisionTransformer architecture (RoPE position
        # embeddings, SwiGLU FFN) at patch_size=14 instead of the public patch16 checkpoints.
        # None of the dinov3 repo's torch.hub entrypoints (dinov3_vitl16 etc.) allow
        # overriding patch_size/ffn_layer, so build the same class directly via its internal
        # _make_dinov3_vit helper. Config below is copied verbatim from the official
        # EncoderConfig in github.com/SMC-GutX/GutCore/blob/main/src/gutcore/config.py
        # (build_encoder() in encoder.py constructs DinoVisionTransformer with these exact
        # kwargs) -- confirmed against the actual checkpoint's tensor shapes: loading it here
        # leaves 0 unexpected_keys and only non-persistent qkv.bias_mask buffers missing.
        torch.hub.list("facebookresearch/dinov3", trust_repo=True)
        dinov3_repo_dir = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov3_main")
        if dinov3_repo_dir not in sys.path:
            sys.path.insert(0, dinov3_repo_dir)
        from dinov3.hub.backbones import _make_dinov3_vit

        backbone = _make_dinov3_vit(
            img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4.0,
            drop_path_rate=0.4,
            layerscale_init=1e-5,
            ffn_layer="swiglu",
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            norm_layer="layernorm",
            n_storage_tokens=0,
            mask_k_bias=False,
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_dtype="bf16",
            compact_arch_name="vitl",
            pretrained=False,
        )
    elif args.model_name.startswith("resnet50"):
        backbone = resnet50()
        backbone.fc = nn.Identity()
    elif args.model_name.startswith("vits16"):
        backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=args.backbone_pretrained_flag,
            num_classes=0
        )
    elif args.model_name.startswith("maxvit_tiny"):
        maxvit_variants = {
            224: "maxvit_tiny_tf_224.in1k",
            384: "maxvit_tiny_tf_384.in1k",
        }
        if args.resize_img_dim not in maxvit_variants:
            raise ValueError(
                f"maxvit_tiny only supports --resize_img_dim 224 or 384, got {args.resize_img_dim}"
            )
        backbone = timm.create_model(
            maxvit_variants[args.resize_img_dim],
            pretrained=args.backbone_pretrained_flag,
            num_classes=0
        )
    elif args.model_name.startswith("mambavision"):
        from transformers import AutoConfig, AutoModel

        class MambaVisionBackbone(nn.Module):
            def __init__(self, pretrained):
                super().__init__()
                repo_id = "nvidia/MambaVision-L-21K"
                if pretrained:
                    self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
                else:
                    hf_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
                    self.model = AutoModel.from_config(hf_config, trust_remote_code=True)

            def forward(self, x):
                out_avg_pool, _ = self.model(x)
                return out_avg_pool

        backbone = MambaVisionBackbone(pretrained=args.backbone_pretrained_flag)
    elif args.model_name.startswith("mamba"):
        from vision_mamba import Vim
        backbone = Vim(
            dim=256,
            heads=8,
            dt_rank=32,
            dim_inner=256,
            d_state=256,
            num_classes=0,
            image_size=224,
            patch_size=16,
            channels=3,
            dropout=0.1,
            depth=12,
        )
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    if args.model_weights:
        if args.model_name.startswith("torch_dinov3_vitl"):
            ckpt_path = hf_hub_download(
                repo_id="tofriede/dinov3-upperGI",
                filename=args.model_weights
            )
            state_dict = torch.load(ckpt_path, map_location=device)

        elif args.model_name.startswith("timm_dinov2_vitb_patch14"):
            state = torch.load(
                str(config.BASE_DIR / "pretrained_weights" / args.model_weights),
                map_location=device
            )
            ini_state_dict = state['teacher']
            state_dict = {k.replace("backbone.", ""): v for k, v in ini_state_dict.items()}

        elif args.model_name.startswith("resnet50"):
            state_dict = torch.load(
                str(config.BASE_DIR / "pretrained_weights" / args.model_weights),
                map_location=device
            )
        elif args.model_name.startswith("torch_dinov3_vits16"):
            state_dict = torch.load(
                str(config.BASE_DIR / "pretrained_weights" / args.model_weights),
                map_location=device
            )
        elif args.model_name.startswith("timm_gutcore_vitl_patch14"):
            state_dict = torch.load(
                str(config.BASE_DIR / "pretrained_weights" / args.model_weights),
                map_location=device
            )
        elif args.model_name.startswith("vits16"):
            state_dict = torch.load(
                str(config.BASE_DIR / "pretrained_weights" / args.model_weights),
                map_location=device
            )
        elif args.model_name.startswith("maxvit_tiny"):
            state_dict = torch.load(
                str(config.BASE_DIR / "pretrained_weights" / args.model_weights),
                map_location=device
            )
        elif args.model_name.startswith("mambavision"):
            state_dict = torch.load(args.model_weights, map_location=device)
        elif args.model_name.startswith("mamba"):
            state_dict = torch.load(args.model_weights, map_location=device)
        else:
            state_dict = None
            print("!!!! No weights loaded !!!!!")

        if state_dict is not None:
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            msg = backbone.load_state_dict(state_dict, strict=False)
            print("missing_keys:", msg.missing_keys)
            print("unexpected_keys:", msg.unexpected_keys)

    if is_vit_backbone(args.model_name) and args.freeze_layers > 0:
        freeze_vit_layers(backbone, args.freeze_layers)
    elif is_resnet_backbone(args.model_name) and args.freeze_layers > 0:
        freeze_resnet_layers(backbone, args.freeze_layers)

    if getattr(args, "use_lora", False):
        apply_lora_to_gutcore_backbone(
            backbone,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(m.strip() for m in args.lora_target_modules.split(",") if m.strip()),
        )
    elif getattr(args, "use_dora", False):
        backbone = apply_dora_to_gutcore_backbone(
            backbone,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=tuple(m.strip() for m in args.lora_target_modules.split(",") if m.strip()),
        )

    return backbone


class BinaryClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_model(args, device):
    backbone = build_backbone(args, device)
    model = BinaryClassifier(backbone, embed_dim=args.backbone_dim, num_classes=1)
    return model.to(device)


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al. 2017, https://arxiv.org/abs/1708.02002)."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_train_loader(train_dataset, train_labels, args):
    num_pos = int(np.sum(np.array(train_labels) == 1))
    num_neg = int(np.sum(np.array(train_labels) == 0))

    print("In train dataset, number of class 0:", num_neg)
    print("In train dataset, number of class 1:", num_pos)

    if args.sampling == "oversample":
        print("Using WeightedRandomSampler for oversampling.")
        class_counts = np.bincount(train_labels, minlength=2)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        sample_weights = [class_weights[label].item() for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )

    elif args.sampling == "undersample":
        print("Using subset undersampling.")
        label_to_subset_indices = defaultdict(list)
        for subset_i, label in enumerate(train_labels):
            label_to_subset_indices[label].append(subset_i)

        min_count = min(len(label_to_subset_indices[0]), len(label_to_subset_indices[1]))
        balanced_subset_indices = random.sample(label_to_subset_indices[0], min_count) + \
                                  random.sample(label_to_subset_indices[1], min_count)

        balanced_dataset = Subset(train_dataset, balanced_subset_indices)

        train_loader = DataLoader(
            balanced_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    else:
        print("Using standard sampling" + (" with class weight." if args.loss == "bce" else "."))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )

    if args.loss == "focal":
        print(f"Using FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma}).")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.sampling == "none":
        pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    return train_loader, criterion, num_pos, num_neg


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al. 2021, https://arxiv.org/abs/2010.01412).

    Wraps a base optimizer (AdamW here) with a two-step update per batch: first_step
    perturbs parameters by an ascent step of radius `rho` toward the direction that
    locally maximizes the loss, then second_step restores the original weights and
    hands off to the base optimizer's real descent step, using gradients computed at
    the perturbed point. Callers must run first_step -> recompute loss/backward ->
    second_step for each batch instead of a single .step() call; adaptive=True gives
    the ASAM variant (Kwon et al. 2021), which scales the perturbation per-parameter
    by |p| instead of using a uniform radius.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # the actual sharpness-aware update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure that reevaluates the loss"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def disable_running_stats(model):
    """Zero out every BatchNorm layer's momentum, ahead of SAM's first (ascent) forward pass.

    SAM runs two forward passes per batch through the same BatchNorm layers; without
    this, running_mean/running_var would get updated twice per batch (once from the
    ascent-step forward, once from the real one), skewing eval-time normalization.
    Momentum is restored by enable_running_stats before the second forward pass.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0


def enable_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum


def build_optimizer_and_scheduler(model, train_loader, args):
    if args.use_sam:
        optimizer = SAM(
            model.parameters(), optim.AdamW, rho=args.sam_rho, adaptive=args.sam_adaptive,
            lr=args.lr, weight_decay=args.weight_decay,
        )
        # LR schedulers monkey-patch optimizer.step to detect "scheduler.step() called
        # before optimizer.step()" -- our training loop never calls SAM.step() directly
        # (it calls first_step/second_step instead), only base_optimizer.step() inside
        # second_step, so the scheduler must be attached to base_optimizer, not to SAM.
        scheduler_optimizer = optimizer.base_optimizer
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_optimizer = optimizer

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.10 * total_steps))

    warmup_scheduler = LinearLR(
        scheduler_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    main_scheduler = CosineAnnealingLR(
        scheduler_optimizer,
        T_max=max(1, total_steps - warmup_steps)
    )

    scheduler = SequentialLR(
        scheduler_optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
    return optimizer, scheduler


def run_one_fold(args, fold_idx, train_indices, val_indices, output_dir, device):
    print(f"\n===== Fold {fold_idx + 1}/{args.n_splits} =====")

    use_extradata_edd2020 = get_edd2020_usage(args) == "train"
    norm_mean, norm_std = get_normalization_stats(args.model_name)
    train_ds = RareDataset(
        args.data_path,
        use_extradata_Endovis=args.use_extradata_Endovis,
        use_extradata_edd2020=use_extradata_edd2020,
        use_extradata_hyper_short_segment=args.use_extradata_hyper_short_segment,
        use_extradata_GastroVision=args.use_extradata_GastroVision,
        use_extradata_synthetic_data=args.use_extradata_synthetic_data,
        use_extradata_barett_archive=args.use_extradata_barett_archive,
        use_extradata_red_patch=args.use_extradata_red_patch,
        transform=get_train_transforms(
            args.resize_img_dim, interpolation=args.interpolation, antialias=args.antialias,
            use_blackbox=args.use_blackbox, use_blur_noise=args.use_blur_noise, use_distortion=args.use_distortion,
            use_zoom=args.use_zoom, use_black_border=args.use_black_border,
            use_zoom_blur_distortion=args.use_zoom_blur_distortion,
            mean=norm_mean, std=norm_std,
        ),
        resize_img_dim=args.resize_img_dim,
    )
    val_ds = RareDataset(
        args.data_path,
        use_extradata_Endovis=args.use_extradata_Endovis,
        use_extradata_edd2020=use_extradata_edd2020,
        use_extradata_hyper_short_segment=args.use_extradata_hyper_short_segment,
        use_extradata_GastroVision=args.use_extradata_GastroVision,
        use_extradata_synthetic_data=args.use_extradata_synthetic_data,
        use_extradata_barett_archive=args.use_extradata_barett_archive,
        use_extradata_red_patch=args.use_extradata_red_patch,
        transform=get_val_transforms(
            args.resize_img_dim, interpolation=args.interpolation, antialias=args.antialias,
            mean=norm_mean, std=norm_std,
        ),
        resize_img_dim=args.resize_img_dim,
    )

    if args.center2_pos_upsample_factor > 1.0:
        train_indices = upsample_group_in_train_indices(
            train_ds, train_indices, factor=args.center2_pos_upsample_factor,
            center_name="center_2", target_label=1, seed=args.seed + fold_idx,
        )

    if args.oversample_filelist and args.oversample_filelist_factor > 1.0:
        oversample_filenames = load_oversample_filelist(args.oversample_filelist)
        train_indices = upsample_filenames_in_train_indices(
            train_ds, train_indices, oversample_filenames,
            factor=args.oversample_filelist_factor, seed=args.seed + fold_idx,
            skip_missing=args.skip_missing_files,
        )

    train_dataset = Subset(train_ds, train_indices)
    val_dataset = Subset(val_ds, val_indices)

    train_labels = [get_sample_label(train_ds.samples[i][1]) for i in train_indices]
    val_labels = [get_sample_label(val_ds.samples[i][1]) for i in val_indices]

    train_num_pos = int(np.sum(np.array(train_labels) == 1))
    train_num_neg = int(np.sum(np.array(train_labels) == 0))
    val_num_pos = int(np.sum(np.array(val_labels) == 1))
    val_num_neg = int(np.sum(np.array(val_labels) == 0))
    print(f"Fold {fold_idx + 1} train class counts: neg={train_num_neg}, pos={train_num_pos}")
    print(f"Fold {fold_idx + 1} val class counts: neg={val_num_neg}, pos={val_num_pos}")

    train_loader, criterion, num_pos, num_neg = build_train_loader(train_dataset, train_labels, args)
    criterion = criterion.to(device)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(args, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, args)

    is_vit_model = (
        args.model_name.startswith("torch_dinov3_vitl")
        or args.model_name.startswith("timm_dinov3_vitl")
        or args.model_name.startswith("timm_gutcore_vitl_patch14")
        or args.model_name.startswith("vits16")
    )
    grad_clip_mode = args.grad_clip_mode if is_vit_model else None
    grad_clip_value = args.grad_clip_value if is_vit_model else None

    fold_model_path = os.path.join(output_dir, f"best_model_fold{fold_idx}.pth")

    best_metric = float("inf") if config.best_model_metric == "Loss" else -float("inf")
    best_epoch = -1
    best_metrics = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            grad_clip_mode=grad_clip_mode, grad_clip_value=grad_clip_value,
            use_sam=args.use_sam,
        )
        val_metrics = validate(model, val_loader, criterion, device)

        current_metric = val_metrics[config.best_model_metric]
        improved = current_metric < best_metric if config.best_model_metric == "Loss" else current_metric > best_metric

        if not improved and args.tiebreak_ppv and best_metrics is not None and current_metric == best_metric:
            improved = val_metrics["PPV"] > best_metrics["PPV"]
            if improved:
                print(
                    f"Fold {fold_idx} | Epoch {epoch + 1}: {config.best_model_metric} tied the "
                    f"current best ({current_metric}) -- tiebreak_ppv selected this epoch "
                    f"(PPV={val_metrics['PPV']} > previous best PPV={best_metrics['PPV']})."
                )

        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_metrics = copy.deepcopy(val_metrics)
            torch.save(model.state_dict(), fold_model_path)
            print(f"Saved best fold model to: {fold_model_path}")

        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val: {val_metrics}"
        )

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_model_path": fold_model_path,
        "best_metric": best_metric,
        "best_metrics": best_metrics,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_train_pos": int(np.sum(np.array(train_labels) == 1)),
        "n_train_neg": int(np.sum(np.array(train_labels) == 0)),
        "n_val_pos": int(np.sum(np.array(val_labels) == 1)),
        "n_val_neg": int(np.sum(np.array(val_labels) == 0)),
    }


def save_cv_results(results, output_csv):
    metric_keys = set()
    for r in results:
        if r["best_metrics"] is not None:
            metric_keys.update(r["best_metrics"].keys())

    metric_keys = sorted(metric_keys)

    with open(output_csv, "w", newline="") as fp:
        writer = csv.writer(fp)
        header = [
            "fold", "best_epoch", "best_model_path", "best_metric",
            "n_train", "n_val", "n_train_pos", "n_train_neg", "n_val_pos", "n_val_neg"
        ] + metric_keys + [f"mean_{m}" for m in metric_keys]
        writer.writerow(header)

        means = {}
        for m in metric_keys:
            vals = [r["best_metrics"][m] for r in results if r["best_metrics"] is not None and m in r["best_metrics"]]
            means[m] = float(np.mean(vals)) if vals else None

        for r in results:
            row = [
                r["fold"], r["best_epoch"], r["best_model_path"], r["best_metric"],
                r["n_train"], r["n_val"], r["n_train_pos"], r["n_train_neg"], r["n_val_pos"], r["n_val_neg"]
            ]
            for m in metric_keys:
                row.append(r["best_metrics"].get(m) if r["best_metrics"] is not None else None)
            for m in metric_keys:
                row.append(means[m])
            writer.writerow(row)

    print(f"Saved CV results to {output_csv}")

def save_holdout_metadata(holdout_filenames, out_path):
    payload = {"holdout_filenames": list(holdout_filenames)}

    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Saved center holdout metadata to {out_path}")


def load_holdout_metadata(in_path):
    with open(in_path, "r") as fp:
        payload = json.load(fp)
    return payload["holdout_filenames"]


def load_folds_metadata(in_path):
    """Load cached fold membership as filenames (not indices).

    Indices from a previous run are only valid against that run's dataset
    composition -- they go stale if files are added/removed (e.g. duplicate
    Endovis/EVCBarrett images deleted) or if --use_extradata_* flags differ.
    Filenames are resolved back to indices fresh via resolve_indices_from_filenames.
    """
    with open(in_path, "r") as fp:
        payload = json.load(fp)

    folds = []
    for f in payload["folds"]:
        folds.append({
            "fold": f["fold"],
            "train_filenames": f["train_filenames"],
            "val_filenames": f["val_filenames"],
        })
    holdout_filenames = payload.get("holdout_filenames", [])
    return folds, holdout_filenames


def main(args):
    print("___________________ K-Fold Ensemble ___________________")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    base_output_dir = '/home/chandraharsha.rachabathuni-umw/Competitions/RARE26_challenge/data_for_modeling/'

    output_dir = os.path.join(
        "/home/chandraharsha.rachabathuni-umw/Competitions/RARE26_challenge/trained_models",
        f"{config.modelname_suffix}"
    )
    os.makedirs(output_dir, exist_ok=True)

    neo_only_active = args.neo_only_other_sources
    if neo_only_active:
        print("neo_only_other_sources enabled: every source folder other than "
              "center_1/center_2/external_edd2020 (Endovis, hyper_short_segment, "
              "GastroVision, synthetic_data, barett_archive, red_patch, external_data_hyper, ...) "
              "is restricted to its 'neo' images only, added to the train set of every "
              "fold and never to validation -- 'ndbe' images from those folders are "
              "dropped entirely. This supersedes --use_extradata_Endovis/"
              "--use_extradata_hyper_short_segment/--use_extradata_GastroVision/"
              "--use_extradata_synthetic_data/--use_extradata_barett_archive/"
              "--use_extradata_red_patch and their *_train_only flags.")

    print("Using Endovis external data: ", 'Yes' if args.use_extradata_Endovis else 'No')

    # neo_only_other_sources handles Endovis placement itself (below), so the normal
    # endovis_train_only path is disabled to avoid double logic.
    endovis_train_only_active = (not neo_only_active) and args.use_extradata_Endovis and args.endovis_train_only
    if endovis_train_only_active:
        print("endovis_train_only enabled: Endovis samples will be excluded from the "
              "stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    edd2020_usage = get_edd2020_usage(args)
    use_extradata_EDD2020 = edd2020_usage == "train"
    edd2020_train_only_active = use_extradata_EDD2020 and args.edd2020_train_only
    print(f"EDD2020 external data usage: {edd2020_usage} "
          f"(train=added to k-fold training data, test=held out and evaluated as an "
          f"ensemble test set, excluded=not used at all)")
    if edd2020_train_only_active:
        print("edd2020_train_only enabled: EDD2020 samples will be excluded from the "
              "stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    print("Using hyper_short_segment external data: ", 'Yes' if args.use_extradata_hyper_short_segment else 'No')

    hyper_short_segment_train_only_active = (not neo_only_active) and args.use_extradata_hyper_short_segment and args.hyper_short_segment_train_only
    if hyper_short_segment_train_only_active:
        print("hyper_short_segment_train_only enabled: hyper_short_segment samples will be excluded "
              "from the stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    print("Using GastroVision external data: ", 'Yes' if args.use_extradata_GastroVision else 'No')

    gastrovision_train_only_active = (not neo_only_active) and args.use_extradata_GastroVision and args.gastrovision_train_only
    if gastrovision_train_only_active:
        print("gastrovision_train_only enabled: GastroVision samples will be excluded "
              "from the stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    print("Using synthetic_data external data: ", 'Yes' if args.use_extradata_synthetic_data else 'No')

    synthetic_data_train_only_active = (not neo_only_active) and args.use_extradata_synthetic_data and args.synthetic_data_train_only
    if synthetic_data_train_only_active:
        print("synthetic_data_train_only enabled: synthetic_data samples will be excluded "
              "from the stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    print("Using barett_archive external data: ", 'Yes' if args.use_extradata_barett_archive else 'No')

    barett_archive_train_only_active = (not neo_only_active) and args.use_extradata_barett_archive and args.barett_archive_train_only
    if barett_archive_train_only_active:
        print("barett_archive_train_only enabled: barett_archive samples will be excluded "
              "from the stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    print("Using red_patch external data: ", 'Yes' if args.use_extradata_red_patch else 'No')

    red_patch_train_only_active = (not neo_only_active) and args.use_extradata_red_patch and args.red_patch_train_only
    if red_patch_train_only_active:
        print("red_patch_train_only enabled: red_patch samples will be excluded "
              "from the stratified fold split (reusing the base fold config) and added to the "
              "train set of every fold only, never to validation.")

    print("Loading base dataset from:", args.data_path)
    base_dataset = RareDataset(
                args.data_path,
                use_extradata_Endovis = args.use_extradata_Endovis,
                use_extradata_edd2020 = use_extradata_EDD2020,
                use_extradata_hyper_short_segment = args.use_extradata_hyper_short_segment,
                use_extradata_GastroVision = args.use_extradata_GastroVision,
                use_extradata_synthetic_data = args.use_extradata_synthetic_data,
                use_extradata_barett_archive = args.use_extradata_barett_archive,
                use_extradata_red_patch = args.use_extradata_red_patch,
                resize_img_dim = args.resize_img_dim)

    # Cache filename reflects the actual composition of the *fold split*, not just
    # whether base_dataset includes the extra data. endovis_train_only leaves Endovis
    # out of the fold split (it's injected into train_idx only, after folds are loaded/
    # created below), so it reuses the base fold file instead of a dedicated one.
    # neo_only_active leaves it out of the fold split the same way (its neo subset is
    # injected into train_idx only, below), so it collapses to '' here too.
    endovis_suffix = 'endo' if (args.use_extradata_Endovis and not endovis_train_only_active and not neo_only_active) else ''

    # Cache filename reflects the actual data composition of base_dataset, not the
    # run's intent -- "test" and "excluded" both leave EDD2020 out of base_dataset,
    # so they share the same cached folds. edd2020_train_only also leaves EDD2020 out
    # of the *fold split* (it's injected into train_idx only, after folds are loaded/
    # created below), so it reuses that same base fold file too.
    edd2020_suffix = 'edd2020' if (edd2020_usage == "train" and not edd2020_train_only_active) else ''

    # Same reasoning as endovis_suffix/edd2020_suffix above: hyper_short_segment_train_only
    # (and neo_only_active) leaves hyper_short_segment out of the *fold split* (injected
    # into train_idx only, after folds are loaded/created below), so it reuses the base
    # fold file too.
    hss_suffix = 'hypershortseg' if (args.use_extradata_hyper_short_segment and not hyper_short_segment_train_only_active and not neo_only_active) else ''

    # Same reasoning as endovis_suffix/edd2020_suffix/hss_suffix above: gastrovision_train_only
    # (and neo_only_active) leaves GastroVision out of the *fold split* (injected into
    # train_idx only, after folds are loaded/created below), so it reuses the base fold
    # file too.
    gastrovision_suffix = 'gastrovision' if (args.use_extradata_GastroVision and not gastrovision_train_only_active and not neo_only_active) else ''

    # Same reasoning as endovis_suffix/edd2020_suffix/hss_suffix/gastrovision_suffix
    # above: synthetic_data_train_only (and neo_only_active) leaves synthetic_data out
    # of the *fold split* (injected into train_idx only, after folds are loaded/created
    # below), so it reuses the base fold file too.
    synthetic_data_suffix = 'syntheticdata' if (args.use_extradata_synthetic_data and not synthetic_data_train_only_active and not neo_only_active) else ''

    # Same reasoning as endovis_suffix/edd2020_suffix/hss_suffix/gastrovision_suffix/
    # synthetic_data_suffix above: barett_archive_train_only (and neo_only_active) leaves
    # barett_archive out of the *fold split* (injected into train_idx only, after folds
    # are loaded/created below), so it reuses the base fold file too.
    barett_archive_suffix = 'barettarchive' if (args.use_extradata_barett_archive and not barett_archive_train_only_active and not neo_only_active) else ''

    # Same reasoning as endovis_suffix/edd2020_suffix/hss_suffix/gastrovision_suffix/
    # synthetic_data_suffix/barett_archive_suffix above: red_patch_train_only (and
    # neo_only_active) leaves red_patch out of the *fold split* (injected into train_idx
    # only, after folds are loaded/created below), so it reuses the base fold file too.
    red_patch_suffix = 'redpatch' if (args.use_extradata_red_patch and not red_patch_train_only_active and not neo_only_active) else ''

    # neo_only_active additionally excludes external_data_hyper (never flag-gated in
    # RareDataset to begin with, so it isn't covered by any suffix above) and drops the
    # 'ndbe' half of every other-source folder entirely -- both changes shift the fold
    # split's composition, so this needs its own cache suffix.
    neo_only_suffix = 'neoonlyother' if neo_only_active else ''

    centersplit_suffix = f"centerholdout{args.center_test_size}" if args.split_centers_train_test else ''

    # --run_tag lets a run opt out of the shared fold cache entirely (e.g. for a
    # one-off experiment that shouldn't read/write the common master_folds file),
    # by giving it its own dedicated fold file.
    run_tag_suffix = args.run_tag if args.run_tag else ''

    master_folds_filename = f"master_folds_base_data_{endovis_suffix}_{edd2020_suffix}_{hss_suffix}_{gastrovision_suffix}_{synthetic_data_suffix}_{barett_archive_suffix}_{red_patch_suffix}_{centersplit_suffix}_{neo_only_suffix}_{run_tag_suffix}.json"

    print("Folds info save to file: ", master_folds_filename)

    shared_folds_path = os.path.join(base_output_dir, master_folds_filename)

    print("Splitting center_1/center_2 into train/test: ", 'Yes' if args.split_centers_train_test else 'No')

    center_holdout_indices = []
    if args.split_centers_train_test:
        center_holdout_path = os.path.join(
            base_output_dir, f"master_center_holdout_{args.center_test_size}.json"
        )
        if os.path.exists(center_holdout_path):
            print(f"Loading existing center holdout split from: {center_holdout_path}")
            center_holdout_filenames = load_holdout_metadata(center_holdout_path)
        else:
            print(f"Creating new center holdout split and saving to: {center_holdout_path}")
            center_holdout_filenames = split_center_holdout(
                base_dataset, test_size=args.center_test_size, seed=args.seed,
            )
            save_holdout_metadata(center_holdout_filenames, center_holdout_path)
        # Filenames are stable across --use_extradata_* flag combinations; indices are not,
        # so resolve indices fresh against this run's base_dataset every time.
        center_holdout_indices = resolve_indices_from_filenames(
            base_dataset, center_holdout_filenames, skip_missing=args.skip_missing_files
        )
        # Save a copy to the model's directory for logging purposes
        save_holdout_metadata(center_holdout_filenames, os.path.join(output_dir, "center_holdout_copy.json"))

    print("____Setting up Folds____")

    endovis_indices = []
    if endovis_train_only_active:
        endovis_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_endovis_sample(img_path)
        ]
        print(f"endovis_train_only: {len(endovis_indices)} Endovis samples excluded from "
              f"the stratified fold split and will be added to train only for every fold.")

    edd2020_indices = []
    if edd2020_train_only_active:
        edd2020_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_edd2020_sample(img_path)
        ]
        print(f"edd2020_train_only: {len(edd2020_indices)} EDD2020 samples excluded from "
              f"the stratified fold split and will be added to train only for every fold.")

    hyper_short_segment_indices = []
    if hyper_short_segment_train_only_active:
        hyper_short_segment_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_hyper_short_segment_sample(img_path)
        ]
        print(f"hyper_short_segment_train_only: {len(hyper_short_segment_indices)} hyper_short_segment "
              f"samples excluded from the stratified fold split and will be added to train only for "
              f"every fold.")

    gastrovision_indices = []
    if gastrovision_train_only_active:
        gastrovision_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_gastrovision_sample(img_path)
        ]
        print(f"gastrovision_train_only: {len(gastrovision_indices)} GastroVision "
              f"samples excluded from the stratified fold split and will be added to train only for "
              f"every fold.")

    synthetic_data_indices = []
    if synthetic_data_train_only_active:
        synthetic_data_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_synthetic_data_sample(img_path)
        ]
        print(f"synthetic_data_train_only: {len(synthetic_data_indices)} synthetic_data "
              f"samples excluded from the stratified fold split and will be added to train only for "
              f"every fold.")

    barett_archive_indices = []
    if barett_archive_train_only_active:
        barett_archive_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_barett_archive_sample(img_path)
        ]
        print(f"barett_archive_train_only: {len(barett_archive_indices)} barett_archive "
              f"samples excluded from the stratified fold split and will be added to train only for "
              f"every fold.")

    red_patch_indices = []
    if red_patch_train_only_active:
        red_patch_indices = [
            i for i, (img_path, _) in enumerate(base_dataset.samples)
            if is_red_patch_sample(img_path)
        ]
        print(f"red_patch_train_only: {len(red_patch_indices)} red_patch "
              f"samples excluded from the stratified fold split and will be added to train only for "
              f"every fold.")

    # neo_only_other_sources supersedes endovis_indices/hyper_short_segment_indices/
    # gastrovision_indices/synthetic_data_indices/barett_archive_indices/red_patch_indices
    # above (all stay empty under neo_only_active, since their
    # *_train_only_active flags are forced off) and additionally covers
    # external_data_hyper, which isn't gated by any --use_extradata_* flag in RareDataset
    # to begin with. Matched by path (is_other_source_sample), not by which flags are
    # set, so every non-center/non-edd2020 folder is covered.
    neo_only_train_indices = []
    neo_only_excluded_ndbe_indices = []
    if neo_only_active:
        for i, (img_path, label) in enumerate(base_dataset.samples):
            if not is_other_source_sample(img_path):
                continue
            if get_sample_label(label) == 1:
                neo_only_train_indices.append(i)
            else:
                neo_only_excluded_ndbe_indices.append(i)
        print(f"neo_only_other_sources: {len(neo_only_train_indices)} 'neo' samples from "
              f"non-center/non-edd2020 source folders excluded from the stratified fold "
              f"split and will be added to train only for every fold; "
              f"{len(neo_only_excluded_ndbe_indices)} 'ndbe' samples from those same "
              f"folders are excluded entirely (never used for train, val, or test).")

    if os.path.exists(shared_folds_path):
        print(f"Loading existing folds from: {shared_folds_path}")
        loaded_folds, loaded_holdout_filenames = load_folds_metadata(shared_folds_path)
        if args.split_centers_train_test and not center_holdout_indices:
            center_holdout_indices = resolve_indices_from_filenames(
                base_dataset, loaded_holdout_filenames, skip_missing=args.skip_missing_files
            )
        # Cached fold membership is stored as filenames for the same reason as the holdout
        # split -- resolve to indices fresh against this run's base_dataset, so deleted
        # files (e.g. dedup'd Endovis/EVCBarrett duplicates) don't crash the run. This base
        # fold file never contains EDD2020 or (when endovis_train_only) Endovis filenames
        # (see edd2020_suffix/endovis_suffix above), so nothing resolves to those indices
        # here even when base_dataset includes those samples.
        folds = []
        for f in loaded_folds:
            folds.append({
                "fold": f["fold"],
                "train_idx": resolve_indices_from_filenames(
                    base_dataset, f["train_filenames"], skip_missing=args.skip_missing_files
                ),
                "val_idx": resolve_indices_from_filenames(
                    base_dataset, f["val_filenames"], skip_missing=args.skip_missing_files
                ),
            })
    else:
        print(f"Creating new folds and saving to: {shared_folds_path}")
        exclude_for_folds = (
            list(center_holdout_indices) + list(edd2020_indices) + list(endovis_indices)
            + list(hyper_short_segment_indices) + list(gastrovision_indices)
            + list(synthetic_data_indices) + list(barett_archive_indices)
            + list(red_patch_indices)
            + list(neo_only_train_indices) + list(neo_only_excluded_ndbe_indices)
        )
        folds = create_folds(
            base_dataset,
            n_splits=args.n_splits,
            seed=args.seed,
            exclude_indices=exclude_for_folds if exclude_for_folds else None,
        )
        # Save to the shared directory for future models to use. Never includes EDD2020 or
        # (when endovis_train_only) Endovis samples (they're excluded above and injected
        # into train_idx in-memory only, below), so this cached file stays the same base
        # fold split regardless of whether this particular run adds them to train.
        save_folds_metadata(base_dataset, folds, shared_folds_path, holdout_indices=center_holdout_indices)

    if endovis_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + endovis_indices

    if edd2020_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + edd2020_indices

    if hyper_short_segment_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + hyper_short_segment_indices

    if gastrovision_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + gastrovision_indices

    if synthetic_data_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + synthetic_data_indices

    if barett_archive_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + barett_archive_indices

    if red_patch_train_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + red_patch_indices

    if neo_only_active:
        for f in folds:
            f["train_idx"] = f["train_idx"] + neo_only_train_indices

    # Save a copy to the model's directory for logging purposes -- reflects the actual
    # training data used for this run, including EDD2020 if edd2020_train_only is set.
    save_folds_metadata(base_dataset, folds, os.path.join(output_dir, "folds_copy.json"),
                         holdout_indices=center_holdout_indices)

    all_results = []
    for fold in folds:
        result = run_one_fold(
            args=args,
            fold_idx=fold["fold"],
            train_indices=fold["train_idx"],
            val_indices=fold["val_idx"],
            output_dir=output_dir,
            device=device
        )
        all_results.append(result)

    save_cv_results(all_results, os.path.join(output_dir, "cv_results.csv"))

    run = None
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            reinit=True,
            name=config.modelname_suffix,
            job_type="summary",
            config={
                **vars(args),
                "n_splits": args.n_splits,
                "data_description": os.path.basename(shared_folds_path),
            },
        )

    print("Master folds file used: ", shared_folds_path)
    print("Experiment name: ", config.modelname_suffix)

    print("\n===== Cross-validation summary =====")
    folds_with_metrics = [r for r in all_results if r["best_metrics"] is not None]
    if len(folds_with_metrics) < len(all_results):
        print(
            f"WARNING: {len(all_results) - len(folds_with_metrics)}/{len(all_results)} "
            "fold(s) never produced a valid (non-NaN) validation metric -- likely "
            "diverged training -- and are excluded from this summary."
        )
    if folds_with_metrics:
        metric_names = sorted(folds_with_metrics[0]["best_metrics"].keys())
        cv_summary_log = {}
        for m in metric_names:
            vals = [r["best_metrics"][m] for r in folds_with_metrics if m in r["best_metrics"]]
            mean_v, std_v = np.mean(vals), np.std(vals)
            print(f"{m}: mean={mean_v:.4f}, std={std_v:.4f}")
            cv_summary_log[f"cv/{m}_mean"] = mean_v
            cv_summary_log[f"cv/{m}_std"] = std_v
        if args.use_wandb:
            wandb.log(cv_summary_log)

    if edd2020_usage == "test":
        print("Loading test dataset for test evaluation.")
        test_norm_mean, test_norm_std = get_normalization_stats(args.model_name)
        test_dataset = RareTestSet(args.data_path_test,
                transform=get_val_transforms(
                    args.resize_img_dim, interpolation=args.interpolation, antialias=args.antialias,
                    mean=test_norm_mean, std=test_norm_std,
                ),
                resize_img_dim = args.resize_img_dim)
        ### Use RARETest dataset as eval dataset for ensemble models prediction and get metrics ###

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        all_true_labels = np.array([
            1 if label == "neoplasia" else 0
            for _, label in test_dataset.samples
        ])

        fold_probs = []
        per_fold_test_metrics = []

        for result in all_results:
            fold_idx = result["fold"]
            model_path = result["best_model_path"]
            print(f"\nEvaluating fold {fold_idx} model: {model_path}")

            model = build_model(args, device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_scores = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    if args.use_tta:
                        probs = predict_with_tta(model, images, args.tta_transforms)
                    else:
                        probs = torch.sigmoid(model(images))
                    fold_scores.extend(probs.cpu().numpy().reshape(-1))

            fold_scores = np.array(fold_scores)
            fold_probs.append(fold_scores)

            fold_test_metrics = round_metrics(compute_metrics(all_true_labels, fold_scores))
            per_fold_test_metrics.append(fold_test_metrics)
            print(f"Fold {fold_idx} test metrics: {fold_test_metrics}")

        ensemble_probs = np.mean(fold_probs, axis=0)
        ensemble_metrics = round_metrics(compute_metrics(all_true_labels, ensemble_probs))

        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        misclassified_idx = np.where(ensemble_preds != all_true_labels)[0]

        misclassified_csv_path = os.path.join(output_dir, "misclassified_test_samples.csv")
        with open(misclassified_csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename", "true_label", "predicted_label", "predicted_prob"])
            for idx in misclassified_idx:
                filename = Path(test_dataset.samples[idx][0]).name
                writer.writerow([
                    filename,
                    int(all_true_labels[idx]),
                    int(ensemble_preds[idx]),
                    float(ensemble_probs[idx]),
                ])
        print(f"Saved {len(misclassified_idx)} misclassified filenames to: {misclassified_csv_path}")

        print(f"Master folds file used: {shared_folds_path}")
        print(f"Experiment name: {config.modelname_suffix}")

        print("\n===== Ensemble Test Metrics =====")
        for k, v in ensemble_metrics.items():
            print(f"  {k}: {v:.4f}")

        if args.use_wandb:
            wandb.log({f"test/ensemble_{k}": v for k, v in ensemble_metrics.items()})
            for k, v in ensemble_metrics.items():
                wandb.run.summary[f"test_ensemble_{k.lower().replace('%', 'pct').replace('@', '_at_').replace(' ', '_')}"] = v

        ensemble_results = {
            "per_fold_test_metrics": per_fold_test_metrics,
            "ensemble_test_metrics": ensemble_metrics,
            "tta_transforms": args.tta_transforms if args.use_tta else None,
        }
        ensemble_results_path = os.path.join(output_dir, "ensemble_test_results.json")
        with open(ensemble_results_path, "w") as fp:
            json.dump(ensemble_results, fp, indent=2)
        print(f"Saved ensemble test results to: {ensemble_results_path}")

    if args.split_centers_train_test and center_holdout_indices:
        print("\nEvaluating ensemble on held-out center_1/center_2 test split.")

        holdout_norm_mean, holdout_norm_std = get_normalization_stats(args.model_name)
        holdout_eval_ds = RareDataset(
            args.data_path,
            use_extradata_Endovis=args.use_extradata_Endovis,
            use_extradata_edd2020=use_extradata_EDD2020,
            use_extradata_hyper_short_segment=args.use_extradata_hyper_short_segment,
            use_extradata_GastroVision=args.use_extradata_GastroVision,
            use_extradata_synthetic_data=args.use_extradata_synthetic_data,
            use_extradata_barett_archive=args.use_extradata_barett_archive,
            use_extradata_red_patch=args.use_extradata_red_patch,
            transform=get_val_transforms(
                args.resize_img_dim, interpolation=args.interpolation, antialias=args.antialias,
                mean=holdout_norm_mean, std=holdout_norm_std,
            ),
            resize_img_dim=args.resize_img_dim,
        )
        holdout_dataset = Subset(holdout_eval_ds, center_holdout_indices)
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        holdout_true_labels = np.array([
            get_sample_label(holdout_eval_ds.samples[i][1]) for i in center_holdout_indices
        ])

        holdout_fold_probs = []
        per_fold_holdout_metrics = []

        for result in all_results:
            fold_idx = result["fold"]
            model_path = result["best_model_path"]
            print(f"\nEvaluating fold {fold_idx} model on center holdout test: {model_path}")

            model = build_model(args, device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            fold_scores = []
            with torch.no_grad():
                for images, _ in holdout_loader:
                    images = images.to(device)
                    if args.use_tta:
                        probs = predict_with_tta(model, images, args.tta_transforms)
                    else:
                        probs = torch.sigmoid(model(images))
                    fold_scores.extend(probs.cpu().numpy().reshape(-1))

            fold_scores = np.array(fold_scores)
            holdout_fold_probs.append(fold_scores)

            fold_holdout_metrics = round_metrics(compute_metrics(holdout_true_labels, fold_scores))
            per_fold_holdout_metrics.append(fold_holdout_metrics)
            print(f"Fold {fold_idx} center holdout metrics: {fold_holdout_metrics}")

        holdout_ensemble_probs = np.mean(holdout_fold_probs, axis=0)
        holdout_ensemble_metrics = round_metrics(compute_metrics(holdout_true_labels, holdout_ensemble_probs))

        holdout_ensemble_preds = (holdout_ensemble_probs > 0.5).astype(int)
        misclassified_holdout_idx = np.where(holdout_ensemble_preds != holdout_true_labels)[0]

        misclassified_holdout_csv_path = os.path.join(output_dir, "misclassified_center_holdout_samples.csv")
        with open(misclassified_holdout_csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename", "true_label", "predicted_label", "predicted_prob"])
            for local_idx in misclassified_holdout_idx:
                global_idx = center_holdout_indices[local_idx]
                filename = Path(holdout_eval_ds.samples[global_idx][0]).name
                writer.writerow([
                    filename,
                    int(holdout_true_labels[local_idx]),
                    int(holdout_ensemble_preds[local_idx]),
                    float(holdout_ensemble_probs[local_idx]),
                ])
        print(f"Saved {len(misclassified_holdout_idx)} misclassified center holdout filenames to: {misclassified_holdout_csv_path}")

        print("\n===== Center Holdout Ensemble Test Metrics =====")
        for k, v in holdout_ensemble_metrics.items():
            print(f"  {k}: {v:.4f}")

        if args.use_wandb:
            wandb.log({f"test/center_holdout_ensemble_{k}": v for k, v in holdout_ensemble_metrics.items()})
            for k, v in holdout_ensemble_metrics.items():
                wandb.run.summary[f"test_center_holdout_ensemble_{k.lower().replace('%', 'pct').replace('@', '_at_').replace(' ', '_')}"] = v

        center_holdout_results = {
            "per_fold_holdout_metrics": per_fold_holdout_metrics,
            "ensemble_holdout_metrics": holdout_ensemble_metrics,
            "tta_transforms": args.tta_transforms if args.use_tta else None,
        }
        center_holdout_results_path = os.path.join(output_dir, "center_holdout_test_results.json")
        with open(center_holdout_results_path, "w") as fp:
            json.dump(center_holdout_results, fp, indent=2)
        print(f"Saved center holdout test results to: {center_holdout_results_path}")

    if args.use_wandb and run is not None:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)  # kept for compatibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="RARE-Challenge")
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional custom string appended to the experiment name "
             "(config.modelname_suffix), which in turn is used for the fold "
             "save path (output_dir under trained_models/) and the wandb run name.",
    )

    parser.add_argument(
        "--sampling",
        type=str,
        choices=["none", "oversample", "undersample"],
        default="none",
    )
    parser.add_argument(
        "--center2_pos_upsample_factor",
        type=float,
        default=1.0,
        help="Upsample factor for center_2 positive (neoplasia) samples within each "
             "fold's training set only (applied before --sampling). 1.0 (default) "
             "disables upsampling. 2.0 duplicates every matching train sample once "
             "(so it appears twice); non-integer values duplicate the integer part "
             "for every matching sample and add one extra copy for a random "
             "fraction of them, so the expected upsample ratio matches the factor. "
             "Validation/test splits are never upsampled.",
    )
    parser.add_argument(
        "--oversample_filelist",
        type=str,
        default="",
        help="Path to a newline-delimited file of filenames (bare filenames or full "
             "paths -- only the basename is matched) to slightly oversample within each "
             "fold's training set only (applied before --sampling, after "
             "--center2_pos_upsample_factor). Blank lines and lines starting with '#' "
             "are ignored. Unset (default) disables this. Validation/test splits are "
             "never upsampled.",
    )
    parser.add_argument(
        "--oversample_filelist_factor",
        type=float,
        default=1.6,
        help="Upsample factor applied to the filenames listed in --oversample_filelist. "
             "Only takes effect when --oversample_filelist is also set. Same semantics "
             "as --center2_pos_upsample_factor: 1.0 disables upsampling, 2.0 duplicates "
             "every matching train sample once, and non-integer values duplicate the "
             "integer part for every matching sample plus one extra copy for a random "
             "fraction of them so the expected ratio matches the factor.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "focal"],
        default="bce",
        help="Loss function for training. 'bce' (default) uses BCEWithLogitsLoss, with "
             "pos_weight from the class ratio when --sampling is 'none'. 'focal' uses "
             "a binary focal loss (Lin et al. 2017) configured via --focal_alpha/"
             "--focal_gamma instead.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for focal loss (only used when --loss focal).",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Weight for the positive class in focal loss (only used when --loss focal).",
    )

    parser.add_argument(
        "--use_sam",
        action="store_true",
        help="Wrap the AdamW optimizer with Sharpness-Aware Minimization (Foret et al. "
             "2021). Each training step becomes two forward/backward passes: an ascent "
             "step of radius --sam_rho to find a nearby worst-case perturbation of the "
             "weights, then the real AdamW update using gradients computed there. "
             "Roughly doubles per-step compute. LR scheduling and --grad_clip_mode/"
             "--grad_clip_value both still apply, and BatchNorm running stats (for "
             "resnet50) are only updated from the second (real) forward pass.",
    )
    parser.add_argument(
        "--sam_rho",
        type=float,
        default=0.05,
        help="Neighborhood size for the SAM ascent step (only used when --use_sam is set).",
    )
    parser.add_argument(
        "--sam_adaptive",
        action="store_true",
        help="Use ASAM (Kwon et al. 2021) instead of vanilla SAM: scales the ascent "
             "perturbation per-parameter by |weight| instead of a uniform radius. Only "
             "used when --use_sam is set.",
    )

    MODEL_NAME_CHOICES = [
    "torch_dinov3_vitl16", "timm_dinov2_vitb_patch14", "timm_dinov3_vitl16", "resnet50",
    "torch_dinov3_vits16", "vits16", "mamba", "mambavision_l2_21k", "maxvit_tiny",
    "timm_gutcore_vitl_patch14",
    ]

    parser.add_argument(
        "--model_name",
        type=str,
        choices=MODEL_NAME_CHOICES,
        required=True,
    )
    parser.add_argument("--model_weights", type=str, default="")
    parser.add_argument("--backbone_pretrained_flag", action="store_true")
    parser.add_argument(
        "--backbone_dim",
        type=int,
        default=1024,
        help="Backbone embedding dimension. Use 640 for mambavision_l2_1k, 384 for vits16, 512 for maxvit_tiny.",
    )
    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=0,
        help="Freeze the first N layers of the backbone, leaving the rest trainable. "
             "For ViT-style --model_name values (torch_dinov3_vitl16/vitb16/vits16, "
             "timm_dinov3_vitl16, vits16), this freezes the patch/token embeddings "
             "and the first N transformer blocks. For resnet50, this freezes the "
             "stem (conv1/bn1) and the first N of the 4 residual stages "
             "(layer1..layer4). It is an error to set this for any other model_name.",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Freeze the entire backbone and finetune it via LoRA (Hu et al. 2021) adapters "
             "instead: trainable low-rank adapters are injected into the attn.qkv, attn.proj, "
             "and mlp.w1/w2/w3 Linear layers of every transformer block (see "
             "--lora_target_modules), while every other backbone parameter (and the task "
             "head, which is unaffected) trains normally. Currently only supported for "
             "--model_name timm_gutcore_vitl_patch14, and mutually exclusive with "
             "--freeze_layers (which --use_lora supersedes -- leave it at 0) and --use_dora.",
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        help="Freeze the entire backbone and finetune it via DoRA (Liu et al. 2024, "
             "https://arxiv.org/abs/2402.09353) adapters instead, using HuggingFace's peft "
             "library (LoraConfig(use_dora=True)) rather than the hand-rolled adapters "
             "--use_lora uses. DoRA decomposes each targeted weight into a magnitude vector "
             "and a direction matrix and applies the low-rank update to the direction only, "
             "which tends to track full finetuning more closely than plain LoRA at the same "
             "rank. Targets the same attn.qkv/attn.proj/mlp.w1/w2/w3 Linear layers as "
             "--use_lora (see --lora_target_modules) and shares its "
             "--lora_rank/--lora_alpha/--lora_dropout hyperparameters. Currently only "
             "supported for --model_name timm_gutcore_vitl_patch14, and mutually exclusive "
             "with --freeze_layers (which --use_dora supersedes -- leave it at 0) and "
             "--use_lora. Requires the `peft` package.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="Adapter rank. Only used when --use_lora or --use_dora is set.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="Adapter scaling numerator (adapter output is scaled by lora_alpha / lora_rank). "
             "Only used when --use_lora or --use_dora is set.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout applied to the input of each adapter before its low-rank projection. "
             "Only used when --use_lora or --use_dora is set.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="qkv,proj,w1,w2,w3",
        help="Comma-separated Linear submodule names within each transformer block's "
             "attn/mlp to attach adapters to (qkv/proj live under block.attn, "
             "w1/w2/w3 under block.mlp -- the GutCore-ViT-L SwiGLU FFN's gate/up/down "
             "projections). Only used when --use_lora or --use_dora is set.",
    )

    parser.add_argument(
        "--grad_clip_mode",
        type=str,
        choices=["none", "global_norm", "value"],
        default="none",
        help="Gradient clipping strategy applied when training ViT models: "
             "'none' disables clipping, 'global_norm' clips by global L2 norm, 'value' clips each gradient element to a fixed threshold.",
    )
    parser.add_argument(
        "--grad_clip_value",
        type=float,
        default=1.0,
        help="Threshold for gradient clipping (max norm for 'global_norm', max abs value for 'value').",
    )

    parser.add_argument("--use_extradata_Endovis", action="store_true") ## additonal data flag
    parser.add_argument(
        "--endovis_train_only",
        action="store_true",
        help="Only applies when --use_extradata_Endovis is set. By default in that case, "
             "Endovis samples are stratified-split across train/val like the rest of the "
             "data, and fold membership is cached under a dedicated '..._endo_...' fold "
             "file. With this flag set, Endovis samples are instead excluded from the "
             "stratified split and added to the train set of every fold only, never to "
             "validation -- so the run reuses the same base fold config "
             "('master_folds_base_data___.json') as an Endovis-excluded run, with Endovis "
             "samples injected into train_idx afterwards.",
    )
    parser.add_argument("--eval_after_training", action="store_true") ## additonal data flag
    parser.add_argument(
        "--exclude_edd2020",
        action="store_true",
        help="Fully exclude the EDD2020 external dataset from this run: not added to "
             "k-fold training data, and not used as a held-out ensemble test set even "
             "if --eval_after_training is also set. Use this to measure model "
             "performance with EDD2020 data removed entirely (ablation).",
    )
    parser.add_argument(
        "--edd2020_train_only",
        action="store_true",
        help="Only applies when EDD2020 is being added to k-fold training data (i.e. "
             "--exclude_edd2020 and --eval_after_training are both unset). By default in "
             "that case, EDD2020 samples are stratified-split across train/val like the "
             "rest of the data, and fold membership is cached under a dedicated "
             "'..._edd2020_...' fold file. With this flag set, EDD2020 samples are instead "
             "excluded from the stratified split and added to the train set of every fold "
             "only, never to validation -- so the run reuses the same base fold config "
             "('master_folds_base_data___.json') as an EDD2020-excluded/held-out run, with "
             "EDD2020 samples injected into train_idx afterwards.",
    )
    parser.add_argument("--use_extradata_hyper_short_segment", action="store_true") ## additonal data flag
    parser.add_argument(
        "--hyper_short_segment_train_only",
        action="store_true",
        help="Only applies when --use_extradata_hyper_short_segment is set. By default in "
             "that case, hyper_short_segment samples (from "
             "/data/RARE26_train_data/hyper_short_segment/) are stratified-split across "
             "train/val like the rest of the data, and fold membership is cached under a "
             "dedicated '..._hypershortseg_...' fold file. With this flag set, "
             "hyper_short_segment samples are instead excluded from the stratified split "
             "and added to the train set of every fold only, never to validation -- so the "
             "run reuses the same base fold config ('master_folds_base_data___.json') as a "
             "hyper_short_segment-excluded run, with hyper_short_segment samples injected "
             "into train_idx afterwards.",
    )
    parser.add_argument("--use_extradata_barett_archive", action="store_true") ## additonal data flag
    parser.add_argument(
        "--barett_archive_train_only",
        action="store_true",
        help="Only applies when --use_extradata_barett_archive is set. By default in "
             "that case, barett_archive samples (from "
             "/data/RARE26_train_data/barett_archive/) are stratified-split across "
             "train/val like the rest of the data, and fold membership is cached under a "
             "dedicated '..._barettarchive_...' fold file. With this flag set, "
             "barett_archive samples are instead excluded from the stratified split "
             "and added to the train set of every fold only, never to validation -- so the "
             "run reuses the same base fold config ('master_folds_base_data___.json') as a "
             "barett_archive-excluded run, with barett_archive samples injected "
             "into train_idx afterwards.",
    )
    parser.add_argument("--use_extradata_GastroVision", action="store_true") ## additonal data flag
    parser.add_argument(
        "--gastrovision_train_only",
        action="store_true",
        help="Only applies when --use_extradata_GastroVision is set. By default in "
             "that case, GastroVision samples (from "
             "/data/RARE26_train_data/GastroVision/) are stratified-split across "
             "train/val like the rest of the data, and fold membership is cached under a "
             "dedicated '..._gastrovision_...' fold file. With this flag set, "
             "GastroVision samples are instead excluded from the stratified split "
             "and added to the train set of every fold only, never to validation -- so the "
             "run reuses the same base fold config ('master_folds_base_data___.json') as a "
             "GastroVision-excluded run, with GastroVision samples injected "
             "into train_idx afterwards.",
    )
    parser.add_argument("--use_extradata_red_patch", action="store_true") ## additonal data flag
    parser.add_argument(
        "--red_patch_train_only",
        action="store_true",
        help="Only applies when --use_extradata_red_patch is set. By default in "
             "that case, red_patch samples (from "
             "/data/RARE26_train_data/red_patch/) are stratified-split across "
             "train/val like the rest of the data, and fold membership is cached under a "
             "dedicated '..._redpatch_...' fold file. With this flag set, "
             "red_patch samples are instead excluded from the stratified split "
             "and added to the train set of every fold only, never to validation -- so the "
             "run reuses the same base fold config ('master_folds_base_data___.json') as a "
             "red_patch-excluded run, with red_patch samples injected "
             "into train_idx afterwards.",
    )
    parser.add_argument("--use_extradata_synthetic_data", action="store_true") ## additonal data flag
    parser.add_argument(
        "--synthetic_data_train_only",
        action="store_true",
        help="Only applies when --use_extradata_synthetic_data is set. By default in "
             "that case, synthetic_data samples (from "
             "/data/RARE26_train_data/synthetic_data/) are stratified-split across "
             "train/val like the rest of the data, and fold membership is cached under a "
             "dedicated '..._syntheticdata_...' fold file. With this flag set, "
             "synthetic_data samples are instead excluded from the stratified split "
             "and added to the train set of every fold only, never to validation -- so the "
             "run reuses the same base fold config ('master_folds_base_data___.json') as a "
             "synthetic_data-excluded run, with synthetic_data samples injected "
             "into train_idx afterwards.",
    )
    parser.add_argument(
        "--neo_only_other_sources",
        action="store_true",
        help="Restrict every source folder other than center_1/center_2/external_edd2020 "
             "(Endovis, hyper_short_segment, GastroVision, barett_archive, red_patch, "
             "synthetic_data, external_data_hyper, and any other such folder) to their "
             "'neo' (neoplasia) images only, added to the train set of every k-fold fold "
             "and never to validation; 'ndbe' images from those same folders are dropped "
             "entirely -- not used for train, validation, or test. Supersedes "
             "--use_extradata_Endovis, --use_extradata_hyper_short_segment, "
             "--use_extradata_GastroVision, --use_extradata_synthetic_data, "
             "--use_extradata_barett_archive, --use_extradata_red_patch, and their "
             "*_train_only flags, all of which are ignored for these sources when this "
             "flag is set. Has no effect on "
             "center_1/center_2 or EDD2020 handling, which stay governed by their own "
             "flags (--split_centers_train_test, --exclude_edd2020/--eval_after_training).",
    )
    parser.add_argument("--resize_img_dim", type=int, default=224)
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"],
        default="bilinear",
        help="Interpolation mode used when resizing images.",
    )
    parser.add_argument(
        "--antialias",
        action="store_true",
        help="Apply antialiasing when resizing images.",
    )
    parser.add_argument(
        "--use_blackbox",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Whether to apply RandomBlackBoxes augmentation in training transforms. "
             "Default is False; pass --use_blackbox true to enable.",
    )
    parser.add_argument(
        "--use_blur_noise",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Whether to apply the albumentations blur/noise OneOf augmentation "
             "(MotionBlur/MedianBlur/GaussianBlur/GaussNoise, p=0.3) in training "
             "transforms. Default is False; pass --use_blur_noise true to enable.",
    )
    parser.add_argument(
        "--use_distortion",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Whether to apply the albumentations distortion OneOf augmentation "
             "(OpticalDistortion/GridDistortion/ElasticTransform, p=0.3) in training "
             "transforms. Default is False; pass --use_distortion true to enable.",
    )
    parser.add_argument(
        "--use_zoom",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Whether to apply RandomZoom augmentation in training transforms: with "
             "probability 0.3, zooms in on the image center by a random factor "
             "between 0%% (no zoom) and 10%% zoom. Default is False; pass "
             "--use_zoom true to enable.",
    )
    parser.add_argument(
        "--use_black_border",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Whether to apply RandomBlackBorder augmentation in training transforms: "
             "with probability 0.3, shrinks the resized image to 90%% of "
             "--resize_img_dim and pads the vacated border with black to restore the "
             "original size. Default is False; pass --use_black_border true to "
             "enable.",
    )
    parser.add_argument(
        "--use_zoom_blur_distortion",
        type=lambda v: v.lower() in ("1", "true", "yes"),
        default=False,
        help="Whether to apply the combined zoom/blur/distortion augmentation in "
             "training transforms: with probability 0.4, applies exactly one of "
             "zoom, blur+noise, or distortion, chosen uniformly among the three. "
             "Unlike --use_zoom/--use_blur_noise/--use_distortion, which each fire "
             "independently, this fires at most one of the three per image. "
             "Default is False; pass --use_zoom_blur_distortion true to enable.",
    )

    parser.add_argument("--data_path_test", type=str, default='/home/chandraharsha.rachabathuni-umw/Competitions/RARE26_challenge/data/RARE26_train_data/external_edd2020/')

    parser.add_argument(
        "--split_centers_train_test",
        action="store_true",
        help="Hold out a stratified test split from center_1/center_2 samples only "
             "(under --data_path). The held-out samples are excluded from k-fold "
             "training and validation entirely and evaluated as an ensemble test set "
             "after training. Any additional data enabled via --use_extradata_* flags "
             "is unaffected and stays available for k-fold/train.",
    )
    parser.add_argument(
        "--center_test_size",
        type=float,
        default=0.25,
        help="Fraction of center_1/center_2 samples held out as the test split "
             "when --split_centers_train_test is set.",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Enable test-time augmentation when evaluating the ensemble on the "
             "EDD2020 test set (--eval_after_training) and/or the center holdout "
             "test split (--split_centers_train_test). Each fold's model predicts "
             "on every view listed in --tta_transforms and the probabilities are "
             "averaged before the models are ensembled across folds. Has no effect "
             "on training or k-fold validation.",
    )
    parser.add_argument(
        "--tta_transforms",
        type=str,
        nargs="+",
        choices=list(TTA_TRANSFORMS.keys()),
        default=["none", "hflip", "vflip"],
        help="Views to average over when --use_tta is set. 'none' is the "
             "unaugmented image; the rest are horizontal/vertical flips and "
             "90/180/270 degree rotations applied to the resized+normalized image.",
    )
    parser.add_argument(
        "--best_model_metric",
        type=str,
        choices=[
            "Loss", "AUROC", "AUPRC", "PPV@90% Recall", "PPV",
            "Accuracy", "Sensitivity", "Specificity",
            "PPV+PPV@90% Recall", "PPV@90% Recall+AUPRC",
        ],
        default="PPV@90% Recall",
        help="Validation metric used to pick the best-epoch checkpoint to save for "
             "each fold. 'PPV+PPV@90% Recall' saves based on the sum of PPV and "
             "PPV@90% Recall; 'PPV@90% Recall+AUPRC' sums PPV@90% Recall and AUPRC "
             "instead -- both are combined criteria covering multiple operating "
             "points/metrics at once. 'Loss' is the only metric where lower is "
             "better; every other choice selects for higher values.",
    )
    parser.add_argument(
        "--tiebreak_ppv",
        action="store_true",
        help="Validation metrics are rounded to 4 decimal places before comparison, so "
             "an epoch can tie the current best --best_model_metric score exactly. By "
             "default such a tie is not an improvement, so the earlier (lower-epoch) "
             "checkpoint stays saved. With this flag set, a tie is broken by PPV: the "
             "tying epoch's model is saved as the new best only if its PPV is higher "
             "than the current best epoch's PPV. Has no effect when --best_model_metric "
             "is 'PPV' itself, since a tie there implies PPV is tied too.",
    )
    parser.add_argument(
        "--skip_missing_files",
        action="store_true",
        help="Cached holdout/fold splits (master_center_holdout_*.json, "
             "master_folds_*.json) are stored as filenames and resolved against the "
             "current data directory on every run. By default, a cached filename that "
             "no longer exists on disk raises an error. Set this flag to drop such "
             "filenames instead -- e.g. after intentionally deleting duplicate images "
             "from external_data_EVCBarrett -- rather than regenerating the splits.",
    )

    args = parser.parse_args()

    if args.neo_only_other_sources:
        # These sources are forced on so RareDataset loads their files at all -- the
        # neo-vs-ndbe filtering and train-only placement happen in main(), keyed off
        # path (is_other_source_sample), not off these flags. external_data_hyper isn't
        # gated by a use_extradata_* flag in RareDataset to begin with, so it's covered
        # by that path-based filtering automatically.
        args.use_extradata_Endovis = True
        args.use_extradata_hyper_short_segment = True
        args.use_extradata_GastroVision = True
        args.use_extradata_synthetic_data = True
        args.use_extradata_barett_archive = True
        args.use_extradata_red_patch = True

    if args.freeze_layers > 0 and not (is_vit_backbone(args.model_name) or is_resnet_backbone(args.model_name)):
        raise ValueError(
            "--freeze_layers is only supported for ViT-style --model_name values "
            f"{VIT_BACKBONE_PREFIXES} and resnet-style values {RESNET_BACKBONE_PREFIXES}; "
            f"got model_name={args.model_name!r}."
        )

    if args.use_lora and args.use_dora:
        raise ValueError(
            "--use_lora and --use_dora are mutually exclusive -- both freeze the entire "
            "backbone and train their own low-rank adapters instead; choose one."
        )

    if args.use_lora:
        if not args.model_name.startswith("timm_gutcore_vitl_patch14"):
            raise ValueError(
                "--use_lora is currently only supported for --model_name "
                f"timm_gutcore_vitl_patch14; got model_name={args.model_name!r}."
            )
        if args.freeze_layers > 0:
            raise ValueError(
                "--use_lora already freezes the entire backbone and trains its own LoRA "
                "adapters instead -- combining it with --freeze_layers is not supported. "
                "Leave --freeze_layers at 0."
            )

    if args.use_dora:
        if not args.model_name.startswith("timm_gutcore_vitl_patch14"):
            raise ValueError(
                "--use_dora is currently only supported for --model_name "
                f"timm_gutcore_vitl_patch14; got model_name={args.model_name!r}."
            )
        if args.freeze_layers > 0:
            raise ValueError(
                "--use_dora already freezes the entire backbone and trains its own DoRA "
                "adapters instead -- combining it with --freeze_layers is not supported. "
                "Leave --freeze_layers at 0."
            )

    config.modelname_suffix = build_modelname_suffix(args)
    config.best_model_metric = args.best_model_metric
    main(args)