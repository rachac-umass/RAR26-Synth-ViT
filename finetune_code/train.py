import os
import csv
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import timm
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Handling datastes
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    Subset,
    Dataset
)
from dataset import RareDataset
from utils import get_train_transforms, get_val_transforms

# Models
from huggingface_hub import hf_hub_download
from torchvision.models import resnet50

# Metrics
from metrics import compute_metrics

from torchvision import transforms

class config:
    train_progress_bar = False
    best_model_metric = 'PPV@90% Recall'
    BASE_DIR = Path(__file__).resolve().parent.parent
    modelname_suffix = 'bbox_4x10x5'
    train_center_names = ("center_2","external_edd2020")
    test_center_names = ("center_1",)


def indices_for_centers(dataset, center_names):
    """Indices of samples whose immediate parent folder (under root_dir) is in center_names."""
    return [
        i for i, (img_path, _) in enumerate(dataset.samples)
        if any(name in Path(img_path).parts for name in center_names)
    ]


VIT_BACKBONE_PREFIXES = ("torch_dinov3_vitl", "timm_dinov2_vitb_patch14", "timm_dinov3_vitl", "vits16")
RESNET_BACKBONE_PREFIXES = ("resnet50",)


def is_vit_backbone(model_name):
    return model_name.startswith(VIT_BACKBONE_PREFIXES)


def is_resnet_backbone(model_name):
    return model_name.startswith(RESNET_BACKBONE_PREFIXES)


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
    higher-level features during finetuning.
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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Use tqdm to wrap your dataloader to add the progress bar
    with tqdm(dataloader, desc="Training", unit="batch", disable  = not config.train_progress_bar) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update the progress bar description with loss and accuracy
            if config.train_progress_bar:
                pbar.set_postfix(loss=total_loss / (pbar.n + 1), accuracy=correct / total)

    # Return average loss and accuracy for the epoch
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["Loss"] = total_loss / len(dataloader)

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU or CPU for this job: ', device)

    # === Initialize dataset and dataloaders ===
    # Center-to-center split: train on center_1 + external_edd2020, validate/test on center_2.
    print("loading data from: ", args.data_path)

    ### Apply train transforms (utils.py) to the train split, val transforms (utils.py) to the test split
    train_ds = RareDataset(
        args.data_path,
        use_extradata_edd2020=True,
        transform=get_train_transforms(
            args.resize_img_dim, interpolation=args.interpolation, antialias=args.antialias,
            use_blackbox=args.use_blackbox, use_blur_noise=args.use_blur_noise, use_distortion=args.use_distortion,
        ),
        resize_img_dim=args.resize_img_dim,
    )
    val_ds = RareDataset(
        args.data_path,
        use_extradata_edd2020=True,
        transform=get_val_transforms(args.resize_img_dim, interpolation=args.interpolation, antialias=args.antialias),
        resize_img_dim=args.resize_img_dim,
    )

    train_indices = indices_for_centers(train_ds, config.train_center_names)
    val_indices = indices_for_centers(val_ds, config.test_center_names)

    print(f"Train centers {config.train_center_names}: {len(train_indices)} samples")
    print(f"Test/val center {config.test_center_names}: {len(val_indices)} samples")

    train_dataset = Subset(train_ds, train_indices)
    val_dataset   = Subset(val_ds, val_indices)

    # Compute pos_weight from training subset indices
    train_labels = [train_ds.samples[i][1] for i in train_dataset.indices]
    num_pos = sum(1 for lbl in train_labels if lbl == "neoplasia")
    num_neg = sum(1 for lbl in train_labels if lbl == "nondysplastic")

    print("In train dataset, number of class 0: ",num_neg)
    print("In train dataset, number of class 1: ",num_pos)

    if args.sampling == "oversample":
        print("Using WeightedRandomSampler for oversampling.")
        class_counts = [
            train_dataset.dataset.class_counts["nondysplastic"],
            train_dataset.dataset.class_counts["neoplasia"],
        ]
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

        # Get the original indices used in the split
        train_indices = (
            train_dataset.indices
            if isinstance(train_dataset, Subset)
            else range(len(train_dataset))
        )

        # Create weights based on label
        sample_weights = []
        for idx in train_indices:
            _, label = train_dataset.dataset[idx]
            sample_weights.append(class_weights[label])

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )

    elif args.sampling == "undersample":
        print("Using subset undersampling.")
        label_to_indices = defaultdict(list)

        # Work with original dataset
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            label_to_indices[label].append(i)

        min_count = min(len(label_to_indices[0]), len(label_to_indices[1]))

        balanced_indices = random.sample(
            label_to_indices[0], min_count
        ) + random.sample(label_to_indices[1], min_count)
        balanced_subset = Subset(train_dataset, balanced_indices)

        train_loader = DataLoader(
            balanced_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    else:
        print("Using standard sampling (no rebalancing).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ###################### Initialize model, loss function, and optimizer ######################
    
    if args.model_name.startswith('torch_dinov3_vitl'):
        backbone = torch.hub.load(
        'facebookresearch/dinov3',   
        'dinov3_vitl16',             # <-- new model name (patch size 16, not 14)
        pretrained= args.backbone_pretrained_flag
        )
    elif args.model_name.startswith('timm_dinov2_vitb_patch14'):
        backbone = timm.create_model(
                "timm/vit_base_patch14_dinov2.lvd142m",
                pretrained=False,
                num_classes=0,       # return features instead of classifier logits
                img_size=336,
            )
    elif args.model_name.startswith('timm_dinov3_vitl'):
        backbone = timm.create_model(
                "vit_large_patch16_dinov3.lvd1689m",
                pretrained=args.backbone_pretrained_flag,   
                num_classes=0       # return features instead of classifier logits
            )
    elif args.model_name.startswith('resnet50'):
        backbone = resnet50()
        backbone.fc = nn.Identity()

    elif args.model_name.startswith('vits16'):
        backbone = timm.create_model(
                "vit_small_patch16_224",
                pretrained=args.backbone_pretrained_flag,
                num_classes=0       # return features instead of classifier logits
            )

    elif args.model_name.startswith('mamba'):
        from vision_mamba import Vim
        backbone =  Vim(
                    dim=256,  # Dimension of the transformer model
                    heads=8,  # Number of attention heads
                    dt_rank=32,  # Rank of the dynamic routing matrix
                    dim_inner=256,  # Inner dimension of the transformer model
                    d_state=256,  # Dimension of the state vector
                    num_classes=0,  # Number of output classes
                    image_size=224,  # Size of the input image
                    patch_size=16,  # Size of each image patch
                    channels=3,  # Number of input channels
                    dropout=0.1,  # Dropout rate
                    depth=12,  # Depth of the transformer model
                )

    print('---- Testing the backbone ----')
    print(type(backbone))
    print(backbone.__class__.__name__)
    x = torch.randn(2, 3, args.resize_img_dim, args.resize_img_dim)  # 2 fake RGB images
    with torch.no_grad():
        y = backbone(x)

    # print("model device:", next(backbone.parameters()))
    print("input shape :", x.shape)
    print("output shape:", y.shape)
    print("output dtype:", y.dtype)
    print("has nan     :", torch.isnan(y).any().item())

    # Load the custom pretrained weights for dino v3 upper GI weights
    if args.model_weights:
        if args.model_name.startswith('torch_dinov3_vitl'):
            ckpt_path = hf_hub_download(
            repo_id="tofriede/dinov3-upperGI",
            filename=args.model_weights)
            state_dict = torch.load(ckpt_path, map_location=device)
        elif args.model_name.startswith('timm_dinov2_vitb_patch14'):
            state = torch.load(str(config.BASE_DIR) + '/pretrained_weights/' + args.model_weights, map_location=device)
            ini_state_dict = state['teacher']
            state_dict = {k.replace("backbone.", ""): v for k, v in ini_state_dict.items()}
        elif args.model_name.startswith('resnet50'):
            state_dict = torch.load(str(config.BASE_DIR) + '/pretrained_weights/' + args.model_weights, map_location=device)
        elif args.model_name.startswith('vits16'):
            state_dict = torch.load(str(config.BASE_DIR) + '/pretrained_weights/' + args.model_weights, map_location=device)
        elif args.model_name.startswith('mamba'):
            state_dict = torch.load(args.model_weights, map_location=device)


        # Handle common checkpoint formats (some save under a 'model' or 'state_dict' key)
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Loading the pre-trained weights
        if args.model_weights:
            msg = backbone.load_state_dict(state_dict, strict=False)
            print(msg.missing_keys, msg.unexpected_keys)

    if is_vit_backbone(args.model_name) and args.freeze_layers > 0:
        freeze_vit_layers(backbone, args.freeze_layers)
    elif is_resnet_backbone(args.model_name) and args.freeze_layers > 0:
        freeze_resnet_layers(backbone, args.freeze_layers)

    # Add classification head 
    class DINOv3Classifier(nn.Module):
        def __init__(self, backbone, embed_dim=args.backbone_dim, num_classes=1):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(embed_dim, num_classes)

        def forward(self, x):
            features = self.backbone(x)  
            return self.head(features)

    model = DINOv3Classifier(backbone, embed_dim=args.backbone_dim, num_classes=1)
    model = model.to(device)

    ### Testing the model ###
    print('---- Testing the model ----')
    x = torch.randn(2, 3, args.resize_img_dim, args.resize_img_dim).to(device)   # 2 fake RGB images
    with torch.no_grad():
        y = model(x)

    print("model device:", next(model.parameters()).device)
    print("input shape :", x.shape)
    print("output shape:", y.shape)
    print("output dtype:", y.dtype)
    print("has nan     :", torch.isnan(y).any().item())

    if args.sampling == 'none':
        print("using class weight")
        criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([num_neg/num_pos],dtype = torch.float32).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    

    ### Optimizer ###
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader)
    print('Total steps: ', total_steps)
    warmup_steps = 0.10 * total_steps
    print('Total warm up steps : ', warmup_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )
    
    ##################### WanDB for logging #####################
    if args.use_wandb:
        WANDB_API_KEY = "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"

        # Log in programmatically without prompts
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            project=args.wandb_project,
            config={
                "experiment_code": config.modelname_suffix,
                **vars(args),
                "device": str(device),
                "Metric for Best Model": config.best_model_metric,
                "optimizer": "AdamW",
                "weight_decay": args.weight_decay,
                "loss_fn": "BCEWithLogitsLoss",

                "transforms": {
                    "train": str(get_train_transforms),
                    "val": str(get_val_transforms),
                    "use_blackbox": args.use_blackbox,
                    "use_blur_noise": args.use_blur_noise,
                    "use_distortion": args.use_distortion,
                },
            },
            name=f"train_{args.model_weights.split('.')[0]}",
        )

    ### Training loop ####
    best_loss = 100000 if config.best_model_metric == 'Loss' else 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_metrics["Loss"],
                "val/auroc": val_metrics["AUROC"],
                "val/auprc": val_metrics["AUPRC"],
                "val/ppv90_recall": val_metrics["PPV@90% Recall"],
                "val/accuracy": val_metrics["Accuracy"],
                "val/sensitivity": val_metrics["Sensitivity"],
                "val/specificity": val_metrics["Specificity"],
                "best_val_loss_so_far": best_loss,
                "model/model_weights": args.model_weights,
                "model/backbone_name": "dinov3_vitl16",
            })

        if val_metrics[config.best_model_metric] > best_loss:
            print()
            best_loss = val_metrics[config.best_model_metric]
            print(f"New best {config.best_model_metric} at epoch {epoch + 1} (not saved to disk)")

            if args.use_wandb:
                wandb.run.summary['best_model_choose_metric'] = config.best_model_metric
                wandb.run.summary["best_epoch"] = epoch + 1
                wandb.run.summary["best_val_loss"] = val_metrics["Loss"]
                wandb.run.summary["best_val_auroc"] = val_metrics["AUROC"]
                wandb.run.summary["best_val_auprc"] = val_metrics["AUPRC"]
                wandb.run.summary["best_val_accuracy"] = val_metrics["Accuracy"]
                wandb.run.summary["model_weights"] = args.model_weights
                wandb.run.summary["backbone_name"] = "dinov3_vitl16"

        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Metrics: {val_metrics}"
        )

    print("\n" + "=" * 60)
    print(f"Model name: {args.model_name}")
    print(f"Model weights: {args.model_weights or 'none (trained from scratch/backbone init)'}")
    print(
        "Augmentations used in training: "
        f"use_blackbox={args.use_blackbox}, "
        f"use_blur_noise={args.use_blur_noise}, "
        f"use_distortion={args.use_distortion}"
    )

    print(config.modelname_suffix)
    print("=" * 60)
    print(f"Test metrics (center_2, final epoch {args.epochs}):")

    import pprint
    pprint.pprint(val_metrics)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset root"
    )

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=2e-4, help="Weight decay for AdamW optimizer")

    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="RARE-Challenge", help="WandB project name"
    )
    parser.add_argument(
        "--resize_img_dim", type=int, default=224, help="Image resize dimension for transforms"
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"],
        help="Interpolation mode used when resizing images",
    )
    parser.add_argument(
        "--antialias", action="store_true", default=True, help="Use antialiasing when resizing images"
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
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["none", "oversample", "undersample"],
        default="none",
        help="Sampling strategy to handle class imbalance",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["torch_dinov3_vitl16", "timm_dinov2_vitb_patch14", "timm_dinov3_vitl16", "resnet50", "vits16"],
        # default="torch_dinov3_vitl16",
        help="Backbone model to use",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="",
        help="Checkpoint filename or path for pretrained weights",
    )
    parser.add_argument(
        "--backbone_pretrained_flag",
        action="store_true",
        help="Load default pretrained weights for the selected backbone",
    )
    parser.add_argument(
        "--backbone_dim",
        type=int,
        default=1024,
        help="Backbone feature dimension for classification head. Use 1024 for "
             "torch_dinov3_vitl16/timm_dinov3_vitl16, 2048 for resnet50, 384 for vits16, "
             "768 for timm_dinov2_vitb_patch14.",
    )
    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=0,
        help="Freeze the first N layers of the backbone, leaving the rest trainable. "
             "For ViT-style --model_name values (torch_dinov3_vitl16, timm_dinov2_vitb_patch14, "
             "timm_dinov3_vitl16, vits16), this freezes the patch/token embeddings and the first N transformer "
             "blocks. For resnet50, this freezes the stem (conv1/bn1) and the first N of "
             "the 4 residual stages (layer1..layer4). It is an error to set this for any "
             "other model_name.",
    )

    args = parser.parse_args()

    if args.freeze_layers > 0 and not (is_vit_backbone(args.model_name) or is_resnet_backbone(args.model_name)):
        raise ValueError(
            "--freeze_layers is only supported for ViT-style --model_name values "
            f"{VIT_BACKBONE_PREFIXES} and resnet-style values {RESNET_BACKBONE_PREFIXES}; "
            f"got model_name={args.model_name!r}."
        )

    main(args)