"""Controlled comparison of fresh regression and classification counting heads.

Both heads see identical frozen DeepFreq features, training examples, noise,
batch order, and maximum epoch budget. Validation selects one epoch per head.
"""

from pathlib import Path
import copy
import csv
import json

import numpy as np
import torch
from torch import nn

from data import data, fr, loss, noise
import util


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "fair_counter_head_comparison"
SEED = 20261017
N_TRAIN = 10_000
N_VALID = 2_000
BATCH_SIZE = 128
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
SNR_LINEAR = 1.0
SNRS_DB = (10, 20)


def make_examples(count):
    signals, frequencies, labels = data.gen_signal(
        count, 50, 10, 1.0, distance="normal",
        amplitude="normal_floor", floor_amplitude=0.1,
        variable_num_freq=True,
    )
    noisy = noise.noise_torch(
        torch.from_numpy(signals), SNR_LINEAR, "gaussian_blind"
    )
    return noisy, torch.from_numpy(labels).long(), frequencies


def extract_features(fr_model, old_fc, signals):
    chunks = []
    with torch.no_grad():
        for batch in signals.split(BATCH_SIZE):
            representation = fr_model(batch)
            features = old_fc.mod(representation[:, None]).flatten(start_dim=1)
            chunks.append(features[:, :old_fc.out_layer.in_features])
    return torch.cat(chunks)


def train_heads(train_x, train_y, valid_x, valid_y):
    heads = {}
    best = {}
    history = []
    for name, outputs in (("regression", 1), ("classification", 10)):
        torch.manual_seed(SEED + 1)
        head = nn.Linear(train_x.shape[1], outputs)
        heads[name] = head
        best[name] = {"accuracy": -1.0, "epoch": 0, "state": None}
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
                  for name, head in heads.items()}

    for epoch in range(1, MAX_EPOCHS + 1):
        # Both heads receive the same examples in the same order each epoch.
        order = torch.randperm(len(train_y), generator=torch.Generator().manual_seed(SEED + epoch))
        for name, head in heads.items():
            head.train()
            for indices in order.split(BATCH_SIZE):
                output = head(train_x[indices])
                if name == "regression":
                    error = nn.functional.mse_loss(output[:, 0], train_y[indices].float())
                else:
                    error = nn.functional.cross_entropy(output, train_y[indices] - 1)
                optimizers[name].zero_grad()
                error.backward()
                optimizers[name].step()

            head.eval()
            with torch.no_grad():
                output = head(valid_x)
                prediction = (torch.round(output[:, 0]).long() if name == "regression"
                              else output.argmax(dim=1) + 1)
                accuracy = 100 * (prediction == valid_y).float().mean().item()
            history.append({"epoch": epoch, "head": name, "validation_accuracy_percent": accuracy})
            if accuracy > best[name]["accuracy"]:
                best[name] = {"accuracy": accuracy, "epoch": epoch,
                              "state": copy.deepcopy(head.state_dict())}
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d}: regression {history[-2]['validation_accuracy_percent']:.1f}% | "
                  f"classification {history[-1]['validation_accuracy_percent']:.1f}%", flush=True)

    for name, head in heads.items():
        head.load_state_dict(best[name]["state"])
        head.eval()
    return heads, best, history


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    cpu = torch.device("cpu")
    fr_model = util.load(str(ROOT / "checkpoint/finetune_experiment/fr/epoch_310.pth"), "fr", cpu)[0].eval()
    old_fc = util.load(str(ROOT / "checkpoint/finetune_experiment/fc/epoch_310.pth"), "fc", cpu)[0].eval()
    print("Generating independent training and validation signals", flush=True)
    train_signals, train_y, _ = make_examples(N_TRAIN)
    valid_signals, valid_y, _ = make_examples(N_VALID)
    print("Computing shared frozen features", flush=True)
    train_x = extract_features(fr_model, old_fc, train_signals)
    valid_x = extract_features(fr_model, old_fc, valid_signals)
    del train_signals, valid_signals
    print(f"Features: {train_x.shape[1]} per signal; training both fresh heads", flush=True)
    heads, best, history = train_heads(train_x, train_y, valid_x, valid_y)

    targets = np.load(ROOT / "test_dataset_low_snr/f.npy")
    true_counts = np.sum(targets >= -0.5, axis=1)
    grid = np.linspace(-0.5, 0.5, fr_model.fr_size, endpoint=False)
    rows = []
    for snr_db in SNRS_DB:
        signals = torch.from_numpy(np.load(
            ROOT / f"test_dataset_low_snr/{float(snr_db)}dB.npy"
        )).float()
        test_x = extract_features(fr_model, old_fc, signals)
        with torch.no_grad():
            representation = torch.cat([fr_model(batch) for batch in signals.split(BATCH_SIZE)]).numpy()
        for name, head in heads.items():
            with torch.no_grad():
                output = head(test_x)
                counts = (torch.round(output[:, 0]).long() if name == "regression"
                          else output.argmax(dim=1) + 1)
            counts = counts.clamp(1, 10).numpy()
            estimates = fr.find_freq(representation, counts, grid, max_freq=10)
            rows.append({
                "head": name,
                "test_snr_db": snr_db,
                "selected_epoch": best[name]["epoch"],
                "count_accuracy_percent": 100 * np.mean(counts == true_counts),
                "chamfer_error": loss.chamfer(estimates.copy(), targets.copy()) / len(targets),
            })

    with (OUTPUT / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "validation_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (OUTPUT / "config.json").write_text(json.dumps({
        "seed": SEED, "n_train": N_TRAIN, "n_valid": N_VALID,
        "max_epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE,
        "noise": "gaussian_blind", "snr_linear": SNR_LINEAR,
        "feature_source": "fine-tuned FR and FC body, epoch 310",
    }, indent=2))
    for name, head in heads.items():
        torch.save(head.state_dict(), OUTPUT / f"{name}_head.pt")
    for row in rows:
        print(f"{row['head']:14s} {row['test_snr_db']:2d} dB: "
              f"count accuracy {row['count_accuracy_percent']:.1f}%, "
              f"Chamfer {row['chamfer_error']:.5f}", flush=True)
    print(f"Saved comparison to {OUTPUT}")


if __name__ == "__main__":
    main()
