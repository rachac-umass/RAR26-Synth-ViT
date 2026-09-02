"""Fine-tune MedGemma-4B (multimodal) as a binary Yes/No neoplasia classifier.

Uses the same image data and directory layout as train_ensemble.py (RareDataset,
center_1/center_2/external_data_* folders) but trains a single train/test split
instead of a k-fold ensemble. The model is a Gemma3-based vision-language model
(MedGemma) fine-tuned with QLoRA: the frozen base weights are loaded in 4-bit
NF4 (bitsandbytes) and only small LoRA adapters on the language-model attention
and MLP projections are trained in bf16 -- this quantized-base + trained-adapter
setup is what makes QLoRA "quantization aware" (the model is fine-tuned directly
against its quantized weights, rather than quantizing after a full fp16 fine-tune).

The model is prompted with the image and a fixed Yes/No question. At inference
time we do NOT sample/generate text -- we take a single forward pass, read the
next-token logits at the position right after the prompt, and softmax over just
the "Yes" and "No" token ids to get a calibrated probability of neoplasia. This
keeps the output directly comparable (AUROC/AUPRC/PPV@90%Recall/etc via
metrics.compute_metrics) to the vision-backbone classifiers in train_ensemble.py.

Requires: transformers (with MedGemma/Gemma3 support), peft, bitsandbytes, accelerate.

Example:
    python train_decoder_llm.py \
        --data_path /path/to/RARE26_train_data \
        --split_centers_train_test \
        --epochs 3 --batch_size 2 --grad_accum_steps 8 --use_wandb
"""

import os
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import load_peft_weights

from dataset import RareDataset, RareTestSet
from metrics import compute_metrics
from train_ensemble import (
    get_sample_label,
    get_labels_from_dataset,
    get_edd2020_usage,
    split_center_holdout,
    resolve_indices_from_filenames,
    save_holdout_metadata,
    load_holdout_metadata,
)


class config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    best_model_metric = "PPV@90% Recall"
    train_progress_bar = True
    modelname_suffix = None  # set at runtime by build_modelname_suffix(args)


def build_modelname_suffix(args):
    parts = ["medgemma4b_qlora", Path(args.model_id).name]
    parts.append(f"lora_r{args.lora_r}_a{args.lora_alpha}")
    parts.append(f"{args.resize_img_dim}px")
    parts.append("endo" if args.use_extradata_Endovis else "no_endo")
    parts.append(f"edd2020_{get_edd2020_usage(args)}")

    if args.split_centers_train_test:
        parts.append(f"centerholdout{args.center_test_size}")

    parts.append(args.run_tag)
    return "_".join(parts)


PROMPT = (
    """
    You are an expert gastrointestinal endoscopist evaluating endoscopic images of Barrett’s esophagus.

Your task is to identify whether the image indicates early Barrett esophagus neoplasia .

Compare any focal abnormality with the adjacent Barrett mucosa. Look for:
- Focal nodularity or mucosal thickening
- Slight elevation, plaque-like change, or sessile growth
- Flat but clearly demarcated abnormal mucosa
- Depression, central dimpling, or ulceration
- Focal color change, including redness, pallor, darkness, or heterogeneous coloration
- Irregular surface/mucosal pattern
- Irregular, dilated, tortuous, or disorganized vascular pattern
- Abrupt change in surface or vascular pattern at the lesion border

If a suspicious lesion is present, assign its Paris morphology:
- 0-I: protruding/polypoid or sessile lesion
- 0-IIa: slightly elevated lesion
- 0-IIb: flat lesion with abnormal color, surface, or vascular pattern
- 0-IIc: slightly depressed lesion
- 0-IIa+c: slightly elevated lesion with a central depressed component
- 0-III: excavated or ulcerated lesion

Interpretation guidance:
- 0-IIa and 0-IIb lesions may represent early neoplasia despite subtle morphology.
- A depressed component (0-IIc or 0-IIa+c), ulceration (0-III), or bulky sessile appearance raises concern for deeper invasion.
- Do not diagnose cancer solely from this image. Report visual suspicion and morphology; histology is required for confirmation.

Return the result as only "yes" or "no" and no other text in output.
"""
)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_messages(image, answer=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class MedGemmaYesNoDataset(Dataset):
    """Wraps (img_path, label) samples (from RareDataset/RareTestSet) for MedGemma SFT.

    Images are resized to resize_img_dim x resize_img_dim (default 224, matching
    the vision-backbone pipeline in train_ensemble.py/utils.py) before being
    handed to the MedGemma processor. Note: MedGemma's own image processor will
    still resize this to its pretrained SigLIP input resolution (896x896 for
    medgemma-4b) internally -- the vision tower's positional embeddings and
    token-pooling ratio are fixed at that native resolution, so the model
    cannot actually run at 224x224 without breaking its architecture. This
    pre-resize standardizes/caps the source image fed into that pipeline
    (matching the other models' preprocessing) rather than changing MedGemma's
    true input resolution.
    """

    def __init__(self, samples, resize_img_dim=224):
        self.samples = samples  # list[(Path, "neoplasia"|"nondysplastic")]
        self.resize_img_dim = resize_img_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.resize_img_dim, self.resize_img_dim), Image.BICUBIC)
        label_bin = get_sample_label(label)
        answer = "Yes" if label_bin == 1 else "No"
        return {"image": image, "answer": answer, "label": label_bin, "path": str(img_path)}


def get_answer_token_ids(processor):
    """Resolve the exact first-generated-token ids for 'Yes' / 'No' under this chat template."""
    tokenizer = processor.tokenizer
    prompt_only_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=True, add_generation_prompt=True,
        return_dict=False,
    )
    token_ids = {}
    for answer in ("Yes", "No"):
        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": answer},
            ],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )
        token_ids[answer] = full_ids[len(prompt_only_ids)]
    print(f"Resolved answer token ids -- Yes: {token_ids['Yes']}, No: {token_ids['No']}")
    return token_ids["Yes"], token_ids["No"]


def make_collate_fn(processor, train_mode):
    """Batches PIL images + Yes/No text through the MedGemma processor.

    Train mode: right-pads, builds `labels` with the prompt (incl. image
    tokens) and any right-padding masked to -100, so loss is computed only on
    the answer tokens. Eval mode: left-pads a prompt-only (no answer) batch so
    logits[:, -1, :] is always the next-token distribution for every sample.
    Returns CPU tensors -- move to device in the training/eval loop, not here,
    since this runs inside DataLoader worker processes.
    """

    def collate(batch):
        # One-element sublist per example: Gemma3Processor's make_nested_list_of_images
        # treats a *flat* list of images as a single example's multiple images, not
        # one image per example, so a flat list here silently collapses a batch of N
        # images into a batch of 1 and raises "inconsistently sized batches of images
        # (1) and text (N)" as soon as batch_size > 1.
        images = [[item["image"]] for item in batch]
        labels_bin = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        paths = [item["path"] for item in batch]

        if train_mode:
            processor.tokenizer.padding_side = "right"
            full_texts = [
                processor.apply_chat_template(
                    build_messages(item["image"], item["answer"]),
                    tokenize=False, add_generation_prompt=False,
                )
                for item in batch
            ]
            prompt_texts = [
                processor.apply_chat_template(
                    build_messages(item["image"]), tokenize=False, add_generation_prompt=True,
                )
                for item in batch
            ]

            full_inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
            prompt_inputs = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)

            labels = full_inputs["input_ids"].clone()
            labels[full_inputs["attention_mask"] == 0] = -100
            prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)
            for i, plen in enumerate(prompt_lens.tolist()):
                labels[i, :plen] = -100
            full_inputs["labels"] = labels

            return full_inputs, labels_bin, paths

        else:
            processor.tokenizer.padding_side = "left"
            prompt_texts = [
                processor.apply_chat_template(
                    build_messages(item["image"]), tokenize=False, add_generation_prompt=True,
                )
                for item in batch
            ]
            eval_inputs = processor(text=prompt_texts, images=images, return_tensors="pt", padding=True)
            return eval_inputs, labels_bin, paths

    return collate


def build_model(args, device):

    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # These names match the Gemma3 *language model* attention/MLP projections
        # only -- the SigLIP vision tower uses "out_proj"/"fc1"/"fc2", so the
        # vision encoder stays frozen and only the decoder is adapted.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_one_epoch(model, loader, optimizer, scheduler, device, args):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    with tqdm(loader, desc="Training", unit="batch", disable=not config.train_progress_bar) as pbar:
        for step, (inputs, _labels_bin, _paths) in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), args.grad_clip_value
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if config.train_progress_bar:
                pbar.set_postfix(loss=total_loss / (step + 1))

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, yes_id, no_id, device, out_csv_path=None):
    model.eval()
    all_labels, all_probs, all_paths = [], [], []

    for inputs, labels_bin, paths in tqdm(loader, desc="Evaluating", unit="batch",
                                           disable=not config.train_progress_bar):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :].float()

        pair_logits = torch.stack(
            [next_token_logits[:, yes_id], next_token_logits[:, no_id]], dim=-1
        )
        probs_yes = torch.softmax(pair_logits, dim=-1)[:, 0]

        all_labels.extend(labels_bin.tolist())
        all_probs.extend(probs_yes.cpu().tolist())
        all_paths.extend(paths)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    if out_csv_path:
        preds = (all_probs >= 0.5).astype(int)
        with open(out_csv_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename", "true_label", "predicted_label", "predicted_prob_neoplasia"])
            for path, label, pred, prob in zip(all_paths, all_labels, preds, all_probs):
                writer.writerow([Path(path).name, int(label), int(pred), float(prob)])

    return metrics


def save_split_metadata(train_samples, test_samples, out_path):
    payload = {
        "train_filenames": [Path(p).name for p, _ in train_samples],
        "train_labels": [get_sample_label(l) for _, l in train_samples],
        "test_filenames": [Path(p).name for p, _ in test_samples],
        "test_labels": [get_sample_label(l) for _, l in test_samples],
    }
    with open(out_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved train/test split metadata to {out_path}")


def get_train_test_samples(base_dataset, base_output_dir, args):
    """Single train/test split, reusing the same holdout mechanics as train_ensemble.py.

    With --split_centers_train_test, this loads/creates the exact same
    center_1/center_2 holdout cache (master_center_holdout_<size>.json) that
    train_ensemble.py's --split_centers_train_test uses, so the test set is
    literally the same held-out images. Otherwise, falls back to a plain
    stratified train/test split over the whole base_dataset.
    """
    if args.split_centers_train_test:
        center_holdout_path = os.path.join(
            base_output_dir, f"master_center_holdout_{args.center_test_size}.json"
        )
        if os.path.exists(center_holdout_path):
            print(f"Loading existing center holdout split from: {center_holdout_path}")
            holdout_filenames = load_holdout_metadata(center_holdout_path)
        else:
            print(f"Creating new center holdout split and saving to: {center_holdout_path}")
            holdout_filenames = split_center_holdout(
                base_dataset, test_size=args.center_test_size, seed=args.seed,
            )
            save_holdout_metadata(holdout_filenames, center_holdout_path)

        test_indices = resolve_indices_from_filenames(
            base_dataset, holdout_filenames, skip_missing=args.skip_missing_files
        )
        test_idx_set = set(test_indices)
        train_indices = [i for i in range(len(base_dataset.samples)) if i not in test_idx_set]
    else:
        labels = get_labels_from_dataset(base_dataset)
        all_indices = np.arange(len(base_dataset.samples))
        train_indices, test_indices = train_test_split(
            all_indices, test_size=args.test_size, random_state=args.seed, stratify=labels,
        )
        train_indices, test_indices = train_indices.tolist(), test_indices.tolist()

    train_samples = [base_dataset.samples[i] for i in train_indices]
    test_samples = [base_dataset.samples[i] for i in test_indices]
    print(f"Train/test split: {len(train_samples)} train samples, {len(test_samples)} test samples.")
    return train_samples, test_samples


def build_optimizer_and_scheduler(model, num_update_steps_per_epoch, args):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, args.epochs * num_update_steps_per_epoch)
    warmup_steps = max(1, int(0.10 * total_steps))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    return optimizer, scheduler


def main(args):
    print("___________________ MedGemma-4B QLoRA (single train/test split) ___________________")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Experiment name: {config.modelname_suffix}")

    ### Logging in for model access ###
    print("HF AUTH")
    from huggingface_hub import login
    login(token="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")

    base_output_dir = str(config.BASE_DIR / "data_for_modeling")
    output_dir = str(config.BASE_DIR / "trained_models" / config.modelname_suffix)
    os.makedirs(output_dir, exist_ok=True)

    print("Using Endovis external data: ", 'Yes' if args.use_extradata_Endovis else 'No')

    edd2020_usage = get_edd2020_usage(args)
    use_extradata_edd2020 = edd2020_usage == "train"
    print(f"EDD2020 external data usage: {edd2020_usage} "
          f"(train=added to train split, test=held out and evaluated as an "
          f"external test set, excluded=not used at all)")

    print("Loading base dataset from:", args.data_path)
    base_dataset = RareDataset(
        args.data_path,
        use_extradata_Endovis=args.use_extradata_Endovis,
        use_extradata_edd2020=use_extradata_edd2020,
    )

    train_samples, test_samples = get_train_test_samples(base_dataset, base_output_dir, args)
    save_split_metadata(train_samples, test_samples, os.path.join(output_dir, "train_test_split.json"))

    print("Loading processor and model:", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, token=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.save_pretrained(output_dir)

    yes_id, no_id = get_answer_token_ids(processor)

    train_ds = MedGemmaYesNoDataset(train_samples, resize_img_dim=args.resize_img_dim)
    test_ds = MedGemmaYesNoDataset(test_samples, resize_img_dim=args.resize_img_dim)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=make_collate_fn(processor, train_mode=True),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=make_collate_fn(processor, train_mode=False),
    )

    model = build_model(args, device)

    num_update_steps_per_epoch = max(1, len(train_loader) // args.grad_accum_steps)
    optimizer, scheduler = build_optimizer_and_scheduler(model, num_update_steps_per_epoch, args)

    run = None
    if args.use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            reinit=True,
            name=config.modelname_suffix,
            config={**vars(args), "n_train": len(train_samples), "n_test": len(test_samples)},
        )

    best_metric = -float("inf")
    best_epoch = -1
    best_adapter_dir = os.path.join(output_dir, "best_adapter")
    misclassified_csv_path = os.path.join(output_dir, "misclassified_test_samples.csv")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args)
        test_metrics = evaluate(model, test_loader, yes_id, no_id, device)

        current_metric = test_metrics[config.best_model_metric]
        improved = current_metric > best_metric

        if improved:
            best_metric = current_metric
            best_epoch = epoch + 1
            model.save_pretrained(best_adapter_dir)
            evaluate(model, test_loader, yes_id, no_id, device, out_csv_path=misclassified_csv_path)
            print(f"Saved best LoRA adapter to: {best_adapter_dir}")

        print(f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Test: {test_metrics}")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                **{f"test/{k}": v for k, v in test_metrics.items()},
                "best_metric_so_far": best_metric,
            })

    summary = {
        "experiment_name": config.modelname_suffix,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_model_metric_name": config.best_model_metric,
        "best_adapter_path": best_adapter_dir,
        "n_train": len(train_samples),
        "n_test": len(test_samples),
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)
    print("\n===== Training summary =====")
    print(summary)

    if edd2020_usage == "test" and best_epoch > 0:
        print("\nEvaluating best adapter on the held-out EDD2020 external test set.")
        set_peft_model_state_dict(model, load_peft_weights(best_adapter_dir))

        edd2020_test_ds = MedGemmaYesNoDataset(
            RareTestSet(args.data_path_test).samples, resize_img_dim=args.resize_img_dim
        )
        edd2020_loader = DataLoader(
            edd2020_test_ds, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=make_collate_fn(processor, train_mode=False),
        )
        edd2020_metrics = evaluate(
            model, edd2020_loader, yes_id, no_id, device,
            out_csv_path=os.path.join(output_dir, "misclassified_edd2020_samples.csv"),
        )
        print("\n===== EDD2020 External Test Metrics =====")
        for k, v in edd2020_metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(os.path.join(output_dir, "edd2020_test_results.json"), "w") as fp:
            json.dump(edd2020_metrics, fp, indent=2)

        if args.use_wandb:
            wandb.log({f"test/edd2020_{k}": v for k, v in edd2020_metrics.items()})

    if args.use_wandb and run is not None:
        wandb.run.summary["best_epoch"] = best_epoch
        wandb.run.summary["best_metric"] = best_metric
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it")

    ##### IMPRTANT FOR EXPERIMENT DISTINCTION #####
    parser.add_argument("--run_tag", type=str, default="v1_wi", help="Suffix for the output directory name.")
    ###############################################

    parser.add_argument(
        "--resize_img_dim", type=int, default=224,
        help="Images are resized to this x this before being handed to the MedGemma "
             "processor, matching the other backbones' preprocessing. MedGemma's own "
             "processor will still internally resize to its pretrained SigLIP "
             "resolution (896px for medgemma-4b) -- see MedGemmaYesNoDataset docstring.",
    )
    parser.add_argument(
        "--attn_implementation", type=str, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Gemma3's mixed causal/bidirectional (image-token) attention mask requires "
             "'eager' for training in most transformers versions.",
    )

    # QLoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_value", type=float, default=1.0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="RARE-Challenge")

    # Data composition / split -- mirrors train_ensemble.py so the same data and
    # (optionally) the exact same center holdout split are used.
    parser.add_argument("--use_extradata_Endovis", action="store_true")
    parser.add_argument("--eval_after_training", action="store_true",
                         help="Exclude EDD2020 from train/test and evaluate it separately "
                              "as an external test set after training.")
    parser.add_argument("--exclude_edd2020", action="store_true",
                         help="Fully exclude EDD2020 from this run (not trained on, not evaluated).")
    parser.add_argument("--data_path_test", type=str,
                         default=str(config.BASE_DIR / "data" / "RARE26_train_data" / "external_edd2020"))

    parser.add_argument("--split_centers_train_test", action="store_true",
                         help="Hold out a stratified test split from center_1/center_2 samples only, "
                              "using the same cached split as train_ensemble.py's --split_centers_train_test.")
    parser.add_argument("--center_test_size", type=float, default=0.25)
    parser.add_argument("--test_size", type=float, default=0.2,
                         help="Fraction held out for testing when --split_centers_train_test is not set "
                              "(plain stratified split over the whole dataset).")
    parser.add_argument("--skip_missing_files", action="store_true")

    args = parser.parse_args()
    config.modelname_suffix = build_modelname_suffix(args)
    main(args)
