"""Compare pretrained and fine-tuned DeepFreq checkpoints fairly.

Both models receive the exact same saved test signals at each SNR.  The script
reports metrics over the complete test set and plots one representative signal.
"""

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
OUTPUT_DIR = ROOT / "results" / "model_comparison"
SNRS_DB = (10, 20)
EXAMPLE_INDEX = 13

CHECKPOINTS = {
    "Original": (
        ROOT / "pretrained_models/DeepFreq/frequency_representation_module.pth",
        ROOT / "pretrained_models/DeepFreq/frequency_counting_module.pth",
    ),
    "Fine-tuned": (
        ROOT / "checkpoint/finetune_experiment/fr/epoch_310.pth",
        ROOT / "checkpoint/finetune_experiment/fc/epoch_310.pth",
    ),
}


def load_models(fr_path, fc_path):
    device = torch.device("cpu")
    fr_model = util.load(str(fr_path), "fr", device)[0].eval()
    fc_model = util.load(str(fc_path), "fc", device)[0].eval()
    return fr_model, fc_model


def evaluate(fr_model, fc_model, signals, targets, xgrid):
    signal_tensor = torch.from_numpy(signals).float()
    with torch.no_grad():
        representation_tensor = fr_model(signal_tensor)
        raw_counts = fc_model(representation_tensor).view(-1)

    representation = representation_tensor.numpy()
    true_counts = np.sum(targets >= -0.5, axis=1)
    predicted_counts = np.clip(np.rint(raw_counts.numpy()), 1, targets.shape[1]).astype(int)

    # Oracle-count FNR isolates the quality of the FR module.
    oracle_estimates = fr.find_freq(representation, true_counts, xgrid)
    fnr_percent = 100 * loss.fnr(
        oracle_estimates, targets.copy(), signals.shape[-1]
    ) / len(signals)

    # Count accuracy evaluates the FC module.
    count_accuracy = 100 * np.mean(predicted_counts == true_counts)

    # Predicted-count Chamfer error evaluates the complete pipeline.
    predicted_estimates = fr.find_freq(
        representation, predicted_counts, xgrid, max_freq=targets.shape[1]
    )
    chamfer = loss.chamfer(predicted_estimates.copy(), targets.copy()) / len(signals)

    return {
        "representation": representation,
        "raw_counts": raw_counts.numpy(),
        "predicted_counts": predicted_counts,
        "predicted_estimates": predicted_estimates,
        "fnr_percent": fnr_percent,
        "count_accuracy_percent": count_accuracy,
        "chamfer_error": chamfer,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = np.load(DATA_DIR / "f.npy")
    models = {name: load_models(*paths) for name, paths in CHECKPOINTS.items()}
    xgrid = np.linspace(-0.5, 0.5, next(iter(models.values()))[0].fr_size, endpoint=False)

    rows = []
    evaluations = {}
    for snr_db in SNRS_DB:
        signals = np.load(DATA_DIR / f"{float(snr_db)}dB.npy")
        evaluations[snr_db] = {}
        for name, (fr_model, fc_model) in models.items():
            result = evaluate(fr_model, fc_model, signals, targets, xgrid)
            evaluations[snr_db][name] = result
            rows.append({
                "model": name,
                "snr_db": snr_db,
                "oracle_count_fnr_percent": result["fnr_percent"],
                "count_accuracy_percent": result["count_accuracy_percent"],
                "full_pipeline_chamfer_error": result["chamfer_error"],
            })

    with (OUTPUT_DIR / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(len(SNRS_DB), 2, figsize=(13, 7), sharex=True)
    true_freqs = targets[EXAMPLE_INDEX][targets[EXAMPLE_INDEX] >= -0.5]
    colors = {"Original": "tab:blue", "Fine-tuned": "tab:orange"}

    for row_index, snr_db in enumerate(SNRS_DB):
        for column_index, name in enumerate(("Original", "Fine-tuned")):
            ax = axes[row_index, column_index]
            result = evaluations[snr_db][name]
            curve = result["representation"][EXAMPLE_INDEX]
            estimates = result["predicted_estimates"][EXAMPLE_INDEX]
            estimates = estimates[estimates >= -0.5]
            ax.plot(xgrid, curve, color=colors[name], label=f"{name} FR")
            ax.vlines(true_freqs, curve.min(), curve.max(), color="black",
                      linestyle="--", alpha=0.7, label="True frequency")
            ax.vlines(estimates, curve.min(), curve.max(), color="red",
                      alpha=0.55, label="Estimated frequency")
            predicted_count = result["predicted_counts"][EXAMPLE_INDEX]
            ax.set_title(f"{name}, {snr_db} dB — predicted count {predicted_count}")
            ax.set_ylabel("Learned evidence")
            ax.grid(alpha=0.2)

    for ax in axes[-1]:
        ax.set_xlabel("Normalized frequency")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Same test signal (index {EXAMPLE_INDEX}) for both models", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "visual_comparison.png", dpi=160)
    plt.close(fig)

    print("\nFair comparison over all test signals")
    print("(Lower FNR/Chamfer is better; higher count accuracy is better.)\n")
    for row in rows:
        print(
            f"{row['model']:10s} | {row['snr_db']:2d} dB | "
            f"FNR {row['oracle_count_fnr_percent']:6.2f}% | "
            f"count accuracy {row['count_accuracy_percent']:6.2f}% | "
            f"Chamfer {row['full_pipeline_chamfer_error']:.5f}"
        )
    print(f"\nSaved metrics: {OUTPUT_DIR / 'metrics.csv'}")
    print(f"Saved plot:    {OUTPUT_DIR / 'visual_comparison.png'}")


if __name__ == "__main__":
    main()
