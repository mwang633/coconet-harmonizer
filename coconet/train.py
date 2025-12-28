"""
Training script for Coconet.

Train the model on Bach chorales to learn 4-part harmony.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import argparse

from .model import Coconet, create_model
from .data import load_bach_chorales, create_dataloader, BachChoraleDataset


def train_epoch(
    model: Coconet,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for pianoroll, mask, target in tqdm(dataloader, desc="Training"):
        pianoroll = pianoroll.to(device)
        mask = mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(pianoroll, mask)

        # Compute loss only on masked positions
        # Convert target from one-hot to class indices
        target_indices = target.argmax(dim=-1)  # (batch, voices, time)

        # Reshape for cross entropy
        batch, voices, time, pitches = logits.shape
        logits_flat = logits.permute(0, 1, 3, 2).reshape(-1, time)
        target_flat = target_indices.reshape(-1)

        # Actually we need: logits (N, C), target (N)
        logits_ce = logits.permute(0, 1, 3, 2).reshape(-1, pitches)
        target_ce = target_indices.reshape(-1)

        # Mask to only compute loss on unknown positions
        mask_expanded = mask.reshape(-1)
        inverse_mask = 1 - mask_expanded

        if inverse_mask.sum() > 0:
            loss = criterion(logits_ce, target_ce)
            # Weight by inverse mask (only penalize errors on masked positions)
            loss = (loss * inverse_mask).sum() / (inverse_mask.sum() + 1e-8)
        else:
            loss = criterion(logits_ce, target_ce).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    model: Coconet,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for pianoroll, mask, target in tqdm(dataloader, desc="Validation"):
            pianoroll = pianoroll.to(device)
            mask = mask.to(device)
            target = target.to(device)

            logits = model(pianoroll, mask)

            # Compute metrics
            target_indices = target.argmax(dim=-1)
            pred_indices = logits.argmax(dim=-1)

            batch, voices, time, pitches = logits.shape
            logits_ce = logits.permute(0, 1, 3, 2).reshape(-1, pitches)
            target_ce = target_indices.reshape(-1)

            loss = criterion(logits_ce, target_ce).mean()
            total_loss += loss.item()

            # Accuracy on masked positions
            mask_flat = mask.reshape(-1)
            inverse_mask = (1 - mask_flat).bool()

            pred_flat = pred_indices.reshape(-1)
            target_flat = target_indices.reshape(-1)

            if inverse_mask.sum() > 0:
                correct += (pred_flat[inverse_mask] == target_flat[inverse_mask]).sum().item()
                total += inverse_mask.sum().item()

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def train(
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_layers: int = 32,
    num_filters: int = 128,
    segment_length: int = 32,
    checkpoint_dir: str = "./checkpoints",
    data_dir: str = "./data",
    device: str = None,
):
    """Main training function."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create directories
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading Bach chorales...")
    chorales = load_bach_chorales(data_dir=Path(data_dir))
    print(f"Loaded {len(chorales)} chorales")

    # Create datasets
    dataset = BachChoraleDataset(
        chorales=chorales,
        segment_length=segment_length,
        mask_prob=0.5,
    )

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"Train segments: {len(train_dataset)}, Val segments: {len(val_dataset)}")

    # Create model
    model = create_model(
        num_layers=num_layers,
        num_filters=num_filters,
        device=device
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        scheduler.step(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        # Regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

    print("\nTraining complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Coconet on Bach chorales")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--layers", type=int, default=32, help="Number of layers")
    parser.add_argument("--filters", type=int, default=128, help="Number of filters")
    parser.add_argument("--segment-length", type=int, default=32, help="Segment length")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_layers=args.layers,
        num_filters=args.filters,
        segment_length=args.segment_length,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
