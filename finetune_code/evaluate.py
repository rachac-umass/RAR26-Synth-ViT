import csv
import argparse
from pathlib import Path

import timm
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import RareTestSet
from metrics import compute_metrics


def evaluate_and_save(model, dataloader, device, output_csv):
    model.eval()
    all_data = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.numpy()

            for path, label, score in zip(paths, labels, outputs):
                all_data.append((Path(path).name, int(label), float(score)))

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score"])
        writer.writerows(all_data)


def bootstrap_evaluation(
    csv_path,
    n_bootstrap=1000,
    min_neoplasia=1000,
    ndbe_multiplier=100,
    output_dir=None,
    prefix="bootstrap",
):
    df = pd.read_csv(csv_path)
    results = []

    neoplasia = df[df["label"] == 1]
    ndbe = df[df["label"] == 0]

    for _ in range(n_bootstrap):
        neoplasia_sample = neoplasia.sample(n=min_neoplasia, replace=True)
        ndbe_sample = ndbe.sample(n=min_neoplasia * ndbe_multiplier, replace=True)

        sample = pd.concat([neoplasia_sample, ndbe_sample])
        y_true = sample["label"].values
        y_score = sample["score"].values

        metrics = compute_metrics(y_true, y_score)
        results.append(metrics)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)

    # Compute median and 95% CI
    summary_df = metrics_df.describe(percentiles=[0.025, 0.5, 0.975]).loc[
        ["2.5%", "50%", "97.5%"]
    ]

    # Save if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_dir / f"{prefix}_bootstrap_raw.csv", index=False)
        summary_df.to_csv(output_dir / f"{prefix}_bootstrap_summary.csv")

        print(
            f"Saved raw bootstrap metrics to: {output_dir / f'{prefix}_bootstrap_raw.csv'}"
        )
        print(
            f"Saved summary statistics to: {output_dir / f'{prefix}_bootstrap_summary.csv'}"
        )

    return summary_df


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment_path:
        experiment_path = Path(args.experiment_path)

        # Load model path from experiment folder
        if not args.model_path:
            args.model_path = experiment_path / "best_model.pth"

        # Set default output file if not provided
        if not args.output_file:
            args.output_file = experiment_path / "predictions.csv"

        # Optional: read config.txt
        config_path = experiment_path / "config.txt"
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    key, value = line.strip().split(":", 1)
                    key, value = key.strip(), value.strip()
                    if key == "model" and not args.model:
                        args.model = value
                    elif key == "data_path" and not args.data_path:
                        args.data_path = value

    # === Load dataset and model ===
    dataset = RareTestSet(args.data_path, return_paths=True)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = timm.create_model(args.model, pretrained=True, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # === Evaluate and save predictions ===
    evaluate_and_save(model, test_loader, device, args.output_file)
    print(f"Predictions saved to: {args.output_file}")

    # === Bootstrap evaluation ===
    summary = bootstrap_evaluation(
        args.output_file,
        output_dir=experiment_path,
        prefix=args.model,
        n_bootstrap=1000,
        min_neoplasia=1000,
        ndbe_multiplier=1000,
    )
    print("Bootstrap evaluation summary:")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=r"E:\RARE2025_processed_data\test",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default=r"C:\Users\20172619\OneDrive - TU Eindhoven\PhD\jaar 3\RARE2025-Baselines-private\output\2025-04-11_06-53-27_resnet50",
        help="Path to output experiment folder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=r"C:\Users\20172619\OneDrive - TU Eindhoven\PhD\jaar 3\RARE2025-Baselines-private\output\2025-04-11_06-53-27_resnet50\best_model.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model", type=str, default="resnet50", help="Model name (default: resnet50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=r"C:\Users\20172619\OneDrive - TU Eindhoven\PhD\jaar 3\RARE2025-Baselines-private\output\2025-04-11_06-53-27_resnet50\results.csv",
        help="File to save evaluation results",
    )
    args = parser.parse_args()
    main(args)