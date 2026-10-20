"""Compare original, regression-fine-tuned, and classification counters."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import fr, loss
import util


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "test_dataset_low_snr"
OUTPUT_DIR = ROOT / "results" / "counter_architecture_comparison"
SNRS_DB = (10, 20)

MODELS = {
    "Original regression": (
        ROOT / "pretrained_models/DeepFreq/frequency_representation_module.pth",
        ROOT / "pretrained_models/DeepFreq/frequency_counting_module.pth",
    ),
    "Fine-tuned regression": (
        ROOT / "checkpoint/finetune_experiment/fr/epoch_310.pth",
        ROOT / "checkpoint/finetune_experiment/fc/epoch_310.pth",
    ),
    "Classification counter": (
        ROOT / "checkpoint/finetune_experiment/fr/epoch_310.pth",
        ROOT / "checkpoint/classification_counter_experiment/frequency_counting_classifier_epoch_310.pth",
    ),
}


def predict_counts(counter, representation):
    output = counter(representation)
    # A regression counter has one value; the new classifier has 10 logits.
    if output.shape[1] == 1:
        return np.rint(output[:, 0].numpy()).astype(int)
    return output.argmax(dim=1).numpy().astype(int) + 1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(DATA_DIR / "f.npy")
    true_counts = np.sum(targets >= -0.5, axis=1)
    rows = []

    for snr_db in SNRS_DB:
        signals = torch.from_numpy(np.load(DATA_DIR / f"{float(snr_db)}dB.npy")).float()
        for name, (fr_path, fc_path) in MODELS.items():
            fr_model = util.load(str(fr_path), "fr", torch.device("cpu"))[0].eval()
            counter = util.load(str(fc_path), "fc", torch.device("cpu"))[0].eval()
            with torch.no_grad():
                representation = fr_model(signals)
                predicted_counts = np.clip(predict_counts(counter, representation), 1, 10)
            estimates = fr.find_freq(
                representation.numpy(), predicted_counts,
                np.linspace(-0.5, 0.5, fr_model.fr_size, endpoint=False), max_freq=10,
            )
            rows.append({
                "model": name,
                "snr_db": snr_db,
                "count_accuracy_percent": 100 * np.mean(predicted_counts == true_counts),
                "full_pipeline_chamfer_error": loss.chamfer(estimates.copy(), targets.copy()) / len(targets),
            })

    with (OUTPUT_DIR / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in MODELS:
        subset = [row for row in rows if row["model"] == name]
        ax.plot([row["snr_db"] for row in subset],
                [row["count_accuracy_percent"] for row in subset], marker="o", label=name)
    ax.set_xlabel("Test SNR (dB)")
    ax.set_ylabel("Frequency-count accuracy (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "count_accuracy.png", dpi=160)

    print("Count accuracy (higher is better); Chamfer error (lower is better)\n")
    for row in rows:
        print(f"{row['model']:24s} | {row['snr_db']:2d} dB | "
              f"accuracy {row['count_accuracy_percent']:5.1f}% | "
              f"Chamfer {row['full_pipeline_chamfer_error']:.5f}")


if __name__ == "__main__":
    main()
