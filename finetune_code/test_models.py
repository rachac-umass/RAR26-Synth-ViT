"""Load a k-fold ensemble saved by train_ensemble.py and evaluate it on a data folder.

Points at a model directory under trained_models/ (any architecture supported by
train_ensemble.py) and a data folder under data/ (must contain 'neo'/'ndbe'
subfolders, same layout as RareTestSet expects), runs every best_model_fold*.pth
checkpoint found there, and prints per-fold + ensemble metrics.

Example:
    python test_models.py \
        --model_dir /home/.../trained_models/vits16_weights_..._RemoveShiftInLightEDD \
        --data_dir /home/.../data/RARE26_train_data/external_edd2020
"""

import re
import glob
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RareTestSet
from utils import get_val_transforms, INTERPOLATION_MODES
from metrics import compute_metrics
from train_ensemble import (
    MODEL_NAME_CHOICES,
    TTA_TRANSFORMS,
    build_backbone,
    BinaryClassifier,
    predict_with_tta,
)


def detect_model_name(basename):
    for name in sorted(MODEL_NAME_CHOICES, key=len, reverse=True):
        if basename.startswith(name):
            return name
    raise ValueError(
        f"Could not infer --model_name from model directory name {basename!r} "
        f"(expected it to start with one of {MODEL_NAME_CHOICES}). "
        f"Pass --model_name explicitly to override."
    )


def detect_resize_img_dim(basename):
    match = re.search(r"_(\d+)px_", basename)
    if not match:
        raise ValueError(
            f"Could not infer --resize_img_dim from model directory name {basename!r}. "
            f"Pass --resize_img_dim explicitly to override."
        )
    return int(match.group(1))


def detect_interpolation_antialias(basename):
    interpolations = "|".join(re.escape(name) for name in INTERPOLATION_MODES)
    match = re.search(rf"_({interpolations})_(antialias|noantialias)_", basename)
    if not match:
        raise ValueError(
            f"Could not infer --interpolation/--antialias from model directory name "
            f"{basename!r}. Pass both explicitly to override."
        )
    return match.group(1), match.group(2) == "antialias"


def find_fold_checkpoints(model_dir):
    def fold_index(path):
        match = re.search(r"best_model_fold(\d+)\.pth$", path)
        return int(match.group(1))

    paths = glob.glob(str(Path(model_dir) / "best_model_fold*.pth"))
    if not paths:
        raise FileNotFoundError(f"No best_model_fold*.pth checkpoints found in {model_dir}")
    return sorted(paths, key=fold_index)


def load_fold_model(checkpoint_path, backbone_args, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    embed_dim = state_dict["head.weight"].shape[1]

    backbone = build_backbone(backbone_args, device)
    model = BinaryClassifier(backbone, embed_dim=embed_dim, num_classes=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_fold(model, dataloader, device, use_tta, tta_transforms):
    scores = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            if use_tta:
                probs = predict_with_tta(model, images, tta_transforms)
            else:
                probs = torch.sigmoid(model(images))
            scores.extend(probs.cpu().numpy().reshape(-1))
    return np.array(scores)


def print_metrics(title, metrics):
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"  {key}: {float(value):.4f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_dir = Path(args.model_dir)
    basename = model_dir.name

    model_name = args.model_name or detect_model_name(basename)
    resize_img_dim = args.resize_img_dim or detect_resize_img_dim(basename)
    if args.interpolation and args.antialias != "auto":
        interpolation, antialias = args.interpolation, args.antialias == "true"
    else:
        detected_interpolation, detected_antialias = detect_interpolation_antialias(basename)
        interpolation = args.interpolation or detected_interpolation
        antialias = detected_antialias if args.antialias == "auto" else args.antialias == "true"

    print(f"Model dir: {model_dir}")
    print(f"Resolved architecture -> model_name={model_name}, resize_img_dim={resize_img_dim}, "
          f"interpolation={interpolation}, antialias={antialias}")

    backbone_args = SimpleNamespace(
        model_name=model_name,
        model_weights="",
        backbone_pretrained_flag=False,
        freeze_layers=0,
        resize_img_dim=resize_img_dim,
    )

    checkpoint_paths = find_fold_checkpoints(model_dir)
    print(f"Found {len(checkpoint_paths)} fold checkpoints: {[Path(p).name for p in checkpoint_paths]}")

    test_dataset = RareTestSet(
        args.data_dir,
        transform=get_val_transforms(resize_img_dim, interpolation=interpolation, antialias=antialias),
        resize_img_dim=resize_img_dim,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    true_labels = np.array([1 if label == "neoplasia" else 0 for _, label in test_dataset.samples])

    fold_probs = []
    for checkpoint_path in checkpoint_paths:
        print(f"\nEvaluating checkpoint: {checkpoint_path}")
        model = load_fold_model(checkpoint_path, backbone_args, device)
        scores = predict_fold(model, test_loader, device, args.use_tta, args.tta_transforms)
        fold_probs.append(scores)
        print_metrics(f"Fold metrics ({Path(checkpoint_path).name})", compute_metrics(true_labels, scores))

    ensemble_probs = np.mean(fold_probs, axis=0)
    print_metrics("Ensemble metrics", compute_metrics(true_labels, ensemble_probs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model_dir", type=str, required=True,
                         help="Directory containing best_model_fold*.pth checkpoints, "
                              "as saved by train_ensemble.py (any architecture).")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Folder under data/ containing 'neo'/'ndbe' subfolders to evaluate on.")

    parser.add_argument("--model_name", type=str, choices=MODEL_NAME_CHOICES, default=None,
                         help="Override architecture auto-detected from --model_dir's name.")
    parser.add_argument("--resize_img_dim", type=int, default=None,
                         help="Override resize dim auto-detected from --model_dir's name.")
    parser.add_argument("--interpolation", type=str,
                         choices=["nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"],
                         default='bicubic',
                         help="Override interpolation mode auto-detected from --model_dir's name.")
    parser.add_argument("--antialias", type=str, choices=["auto", "true", "false"], default="auto",
                         help="Override antialias flag auto-detected from --model_dir's name.")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--use_tta", action="store_true",
                         help="Average predictions over --tta_transforms views per fold, "
                              "same as train_ensemble.py's --use_tta.")
    parser.add_argument("--tta_transforms", type=str, nargs="+",
                         choices=list(TTA_TRANSFORMS.keys()),
                         default=["none", "hflip", "vflip"])

    args = parser.parse_args()
    main(args)
