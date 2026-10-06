from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Iterable

import torch
from scipy import sparse
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.3
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class MLPClassifier(nn.Module):
    """Three-hidden-layer MLP for tabular Adult Census classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_classes: int = 2,
    ):
        if hidden_size not in {32, 64}:
            raise ValueError("hidden_size must be either 32 or 64")

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT),
        )
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(self.hidden_layers(x))


def build_model(hidden_size: int, input_size: int, num_classes: int = 2) -> MLPClassifier:
    return MLPClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)


def load_processed_data(path: str | Path) -> dict[str, Tensor]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {data_path}")

    with data_path.open("rb") as file:
        data = pickle.load(file)

    try:
        processed = {}
        for name in ("train_features", "val_features", "test_features"):
            features = data[name]
            if sparse.issparse(features):
                features = features.toarray()
            processed[name] = torch.as_tensor(features, dtype=torch.float32)

        for name in ("train_labels", "val_labels", "test_labels"):
            processed[name] = torch.as_tensor(data[name], dtype=torch.long)

        return processed
    except KeyError as error:
        raise ValueError("Processed pickle is missing one or more required datasets") from error


def build_dataloaders(
    processed_data_path: str | Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    processed = load_processed_data(processed_data_path)
    train_features = processed["train_features"]
    train_labels = processed["train_labels"]
    val_features = processed["val_features"]
    val_labels = processed["val_labels"]
    test_features = processed["test_features"]
    test_labels = processed["test_labels"]

    if train_features.shape[1] != val_features.shape[1] or train_features.shape[1] != test_features.shape[1]:
        raise ValueError("Training, validation, and test data have different feature sizes")

    return (
        DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(val_features, val_labels), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False),
        train_features.shape[1],
    )


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    learning_rate: float


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += value * count
        self.count += count

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss_meter.update(loss.item(), inputs.size(0))

            predictions = logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    return loss_meter.average, correct / total


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), inputs.size(0))
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    return loss_meter.average, correct / total


def save_training_results(metrics: Iterable[TrainingMetrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "epoch": metric.epoch,
            "train_loss": metric.train_loss,
            "train_accuracy": metric.train_accuracy,
            "test_loss": metric.test_loss,
            "test_accuracy": metric.test_accuracy,
            "learning_rate": metric.learning_rate,
        }
        for metric in metrics
    ]
    metrics_path = output_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def save_checkpoint(model: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata}
    torch.save(checkpoint, path)


def train_and_evaluate(
    hidden_size: int,
    processed_data_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    output_dir: str,
) -> tuple[MLPClassifier, list[TrainingMetrics]]:
    device_obj = torch.device(device)
    train_loader, val_loader, test_loader, input_size = build_dataloaders(processed_data_path, batch_size)
    model = build_model(hidden_size=hidden_size, input_size=input_size).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=DEFAULT_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    metrics: list[TrainingMetrics] = []
    best_accuracy = 0.0
    best_state = None
    start_time = time()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_fn, device_obj)
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device_obj)
        scheduler.step(val_accuracy)
        metrics.append(
            TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                test_loss=val_loss,
                test_accuracy=val_accuracy,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4%} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4%} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    elapsed = time() - start_time
    print(f"Training finished in {elapsed:.1f} seconds.")

    output_path = Path(output_dir)
    save_training_results(metrics, output_path)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_accuracy = evaluate(model, test_loader, nn.CrossEntropyLoss(), device_obj)
    print(f"Best validation accuracy: {best_accuracy:.4%}")
    print(f"Final test accuracy: {test_accuracy:.4%}")

    save_checkpoint(
        model,
        output_path / "model_checkpoint.pt",
        {
            "hidden_size": hidden_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "input_size": input_size,
            "best_validation_accuracy": best_accuracy,
            "test_accuracy": test_accuracy,
        },
    )
    return model, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a three-hidden-layer classifier on the Adult Census dataset.")
    parser.add_argument("--hidden-size", type=int, choices=(32, 64), default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--processed-data", type=str, default="processed_data.pkl")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--output-dir", type=str, default="./output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("Starting Adult Census training")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")

    train_and_evaluate(
        hidden_size=args.hidden_size,
        processed_data_path=args.processed_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
