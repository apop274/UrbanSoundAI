"""
Train a CNN on UrbanSound8K mel spectrograms (4 classes).
Loads metadata, filters to siren/dog_bark/drilling/engine_idling,
builds spectrograms, splits train/test, trains model, prints accuracy, saves model.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_CLASSES,
    MODELS_DIR,
    DEFAULT_MODEL_NAME,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.dataset import load_and_filter_metadata, UrbanSoundSpectrogramDataset
from src.model import get_model


def main():
    parser = argparse.ArgumentParser(description="Train UrbanSoundAI CNN")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Output model filename (default: urbansound_cnn.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load metadata and filter to 4 classes
    print("Loading UrbanSound8K metadata and filtering to 4 classes...")
    df = load_and_filter_metadata()
    print(f"Total samples: {len(df)}")
    for c in TARGET_CLASSES:
        print(f"  {c}: {(df['class'] == c).sum()}")

    # 2. Build full dataset (spectrograms are computed on the fly in __getitem__)
    dataset = UrbanSoundSpectrogramDataset(df)

    # 3. Train/test split by indices
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["class"]
    )
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 4. Model, loss, optimizer
    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training loop with progress
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  test_acc={acc:.2f}%  (best={best_acc:.2f}%)"
        )

    # 6. Final accuracy and save model
    print(f"\nFinal test accuracy: {best_acc:.2f}%")
    model_path = os.path.join(MODELS_DIR, args.model_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": TARGET_CLASSES,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
