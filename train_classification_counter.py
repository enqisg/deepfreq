"""Train a classification-based frequency-counting module.

This experiment keeps the fine-tuned frequency-representation (FR) model fixed
and replaces only the old regression counter with a 10-class counter.  Class 0
means one frequency, class 9 means ten frequencies.
"""

from argparse import Namespace
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data import dataset, noise
import modules
import util


ROOT = Path(__file__).resolve().parent
FR_CHECKPOINT = ROOT / "checkpoint/finetune_experiment/fr/epoch_310.pth"
FC_CHECKPOINT = ROOT / "checkpoint/finetune_experiment/fc/epoch_310.pth"
OUTPUT_DIR = ROOT / "checkpoint/classification_counter_experiment"

CONFIG = {
    "seed": 20261017,
    "n_training": 10_000,
    "n_validation": 2_000,
    "batch_size": 128,
    "epochs": 10,
    "learning_rate": 1e-4,
    "snr_linear": 1.0,
    "noise": "gaussian_blind",
    "max_num_freq": 10,
}


def make_loader(args, num_samples, batch_size, xgrid, fixed_noise=False):
    common = dict(
        num_samples=num_samples,
        signal_dim=args.signal_dim,
        max_n_freq=CONFIG["max_num_freq"],
        min_sep=args.min_sep,
        distance=args.distance,
        amplitude=args.amplitude,
        floor_amplitude=args.floor_amplitude,
        kernel_type=args.kernel_type,
        kernel_param=args.gaussian_std / args.signal_dim,
        batch_size=batch_size,
        xgrid=xgrid,
    )
    if fixed_noise:
        return dataset.load_dataloader_fixed_noise(
            **common, snr=CONFIG["snr_linear"], noise=CONFIG["noise"]
        )
    return dataset.load_dataloader(**common)


def classify_counts(model, fr_tensor):
    """Map logits to frequency counts in the inclusive range 1..10."""
    return model(fr_tensor).argmax(dim=1) + 1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cpu")

    fr_model, _, _, fr_args, _ = util.load(str(FR_CHECKPOINT), "fr", device)
    fr_model.eval()
    for parameter in fr_model.parameters():
        parameter.requires_grad = False

    # Reuse the fine-tuned counting feature extractor but replace the final
    # 200-to-1 regression head with a new 200-to-10 classification head.
    old_fc, _, _, old_args, old_epoch = util.load(str(FC_CHECKPOINT), "fc", device)
    classification_args = copy.deepcopy(old_args)
    classification_args.fc_module_type = "classification"
    classification_args.max_num_freq = CONFIG["max_num_freq"]
    classification_args.output_dir = str(OUTPUT_DIR)
    fc_model = modules.set_fc_module(classification_args).to(device)
    transferable = {
        key: value for key, value in old_fc.state_dict().items()
        if not key.startswith("out_layer.")
    }
    fc_model.load_state_dict(transferable, strict=False)

    xgrid = np.linspace(-0.5, 0.5, fr_args.fr_size, endpoint=False)
    train_loader = make_loader(fr_args, CONFIG["n_training"], CONFIG["batch_size"], xgrid)
    validation_loader = make_loader(
        fr_args, CONFIG["n_validation"], CONFIG["batch_size"], xgrid, fixed_noise=True
    )
    optimizer = torch.optim.Adam(fc_model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    history = []

    print("Classification-counter experiment")
    print("FR weights: frozen fine-tuned epoch 310")
    print("FC body: initialized from fine-tuned regression counter")
    print("FC head: new 10-class output layer")
    print(f"Training examples: {CONFIG['n_training']}; epochs: {CONFIG['epochs']}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        start = time.time()
        fc_model.train()
        train_correct = train_total = 0
        train_loss = 0.0
        for clean_signal, _, frequencies in train_loader:
            noisy_signal = noise.noise_torch(
                clean_signal, CONFIG["snr_linear"], CONFIG["noise"]
            )
            with torch.no_grad():
                fr_output = fr_model(noisy_signal)
            targets = (frequencies >= -0.5).sum(dim=1).long() - 1
            logits = fc_model(fr_output)
            batch_loss = criterion(logits, targets)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item() * len(targets)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += len(targets)

        fc_model.eval()
        val_correct = val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for noisy_signal, _, _, frequencies in validation_loader:
                fr_output = fr_model(noisy_signal)
                targets = (frequencies >= -0.5).sum(dim=1).long() - 1
                logits = fc_model(fr_output)
                val_loss += criterion(logits, targets).item() * len(targets)
                val_correct += (logits.argmax(dim=1) == targets).sum().item()
                val_total += len(targets)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy_percent": 100 * train_correct / train_total,
            "validation_loss": val_loss / val_total,
            "validation_accuracy_percent": 100 * val_correct / val_total,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']}: "
            f"train acc {row['train_accuracy_percent']:.2f}%, "
            f"validation acc {row['validation_accuracy_percent']:.2f}%, "
            f"time {time.time() - start:.1f}s"
        )

    # Save a util.load-compatible checkpoint, without touching previous runs.
    checkpoint = {
        "epoch": old_epoch + CONFIG["epochs"],
        "model": fc_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "args": classification_args,
    }
    output_path = OUTPUT_DIR / "frequency_counting_classifier_epoch_310.pth"
    torch.save(checkpoint, output_path)
    (OUTPUT_DIR / "training_config.json").write_text(json.dumps(CONFIG, indent=2))
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved classification checkpoint: {output_path}")


if __name__ == "__main__":
    main()
