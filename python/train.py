"""
Train hand position Transformer model.

Input: 21 tokens x 4 features
  - Previous hand (6 tokens): wrist + 5 fingertips
  - Current notes (5 tokens): one per finger slot
  - Lookahead (10 tokens): next notes with fingering

Output: 12 values (wrist x,y + 5 fingertips x,y)
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import HandPositionTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_data(hand='right', val_split=0.2, seed=42):
    """Load hand position training data."""
    data_dir = PROJECT_ROOT / "data"
    data = np.load(data_dir / 'hand_position_data.npz')

    X = data[f'X_{hand}']
    Y = data[f'Y_{hand}']

    if len(X) == 0:
        return None

    print(f"  Loaded {len(X)} samples")

    # Shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X, Y = X[indices], Y[indices]

    # Split
    n_val = int(len(X) * val_split)
    return {
        'X_train': torch.tensor(X[n_val:], dtype=torch.float32),
        'Y_train': torch.tensor(Y[n_val:], dtype=torch.float32),
        'X_val': torch.tensor(X[:n_val], dtype=torch.float32),
        'Y_val': torch.tensor(Y[:n_val], dtype=torch.float32),
    }


def train(hand='right', max_epochs=300, lr=0.001, d_model=64, nhead=4, num_layers=3,
          dim_feedforward=128, patience=50, lr_patience=15, lr_factor=0.5, min_lr=1e-6):
    """Train hand position transformer."""
    print(f"Training {hand} hand position transformer...")

    data = load_data(hand)
    if data is None:
        print(f"  No data for {hand} hand")
        return None, None

    X_train, Y_train = data['X_train'], data['Y_train']
    X_val, Y_val = data['X_val'], data['Y_val']

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    model = HandPositionTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers,
                                     dim_feedforward=dim_feedforward)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: d_model={d_model}, heads={nhead}, layers={num_layers}, ff={dim_feedforward} ({n_params:,} params)")

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    lr_epochs_without_improvement = 0
    current_lr = lr

    for epoch in range(max_epochs):
        # Train
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()

        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, Y_val).item()
        else:
            val_loss = best_val_loss

        # Check for improvement
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            lr_epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            lr_epochs_without_improvement += 1

        # LR scheduling
        if lr_epochs_without_improvement >= lr_patience and current_lr > min_lr:
            current_lr = max(current_lr * lr_factor, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            lr_epochs_without_improvement = 0
            print(f"  Epoch {epoch+1}: Reduced LR to {current_lr:.2e}")

        # Progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: val_loss={val_loss:.6f}, best={best_val_loss:.6f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        errors = (val_pred - Y_val).abs().mean(dim=0)

        print(f"\n  Best val_loss: {best_val_loss:.6f}")
        print(f"  Mean absolute errors:")
        print(f"    Wrist:  x={errors[0]:.4f}, y={errors[1]:.4f}")
        print(f"    Thumb:  x={errors[2]:.4f}, y={errors[3]:.4f}")
        print(f"    Index:  x={errors[4]:.4f}, y={errors[5]:.4f}")
        print(f"    Middle: x={errors[6]:.4f}, y={errors[7]:.4f}")
        print(f"    Ring:   x={errors[8]:.4f}, y={errors[9]:.4f}")
        print(f"    Pinky:  x={errors[10]:.4f}, y={errors[11]:.4f}")

    # Save model
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    model_path = checkpoints_dir / f"hand_position_{hand}.pt"
    torch.save({
        'model_state': best_model_state,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'val_loss': best_val_loss
    }, model_path)
    print(f"\n  Saved: {model_path}")

    return model, best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hand position transformer")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both",
                        help="Which hand to train (default: both)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Max epochs (default: 300)")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Transformer embedding dimension (default: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer layers (default: 3)")
    parser.add_argument("--dim_feedforward", type=int, default=128,
                        help="Feedforward dimension (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (default: 50)")

    args = parser.parse_args()

    hands = ['left', 'right'] if args.hand == 'both' else [args.hand]

    for hand in hands:
        train(hand,
              max_epochs=args.epochs,
              lr=args.lr,
              d_model=args.d_model,
              nhead=args.nhead,
              num_layers=args.num_layers,
              dim_feedforward=args.dim_feedforward,
              patience=args.patience)
        print()
