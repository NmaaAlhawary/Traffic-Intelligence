"""
Multi-Horizon Traffic Flow Forecasting – LSTM Training (Phase 3 §8.3)
Trains a PyTorch LSTM that predicts vehicle demand at:
  - 15 minutes ahead
  - 30 minutes ahead
  - 60 minutes ahead
Uses detector_dataset.csv + signal_timing_log.csv from the Phase 1 sandbox.

Usage:
  python train_multihorizon.py
  python train_multihorizon.py --epochs 60 --hidden 128
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR   = Path(__file__).parent / "data"
MODEL_DIR  = Path(__file__).resolve().parents[1] / "model"
MODEL_PATH = MODEL_DIR / "lstm_multihorizon.pt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN    = 12   # 12 × 15 min = 3 hours of input history
HORIZONS   = [1, 2, 4]   # steps ahead → 15 min, 30 min, 60 min


class TrafficLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, layers: int = 2,
                 n_horizons: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_horizons),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # use last timestep


def load_and_prepare(data_dir: Path):
    det_path = data_dir / "detector_dataset.csv"
    sig_path = data_dir / "signal_timing_log.csv"

    if not det_path.exists():
        raise FileNotFoundError(
            f"{det_path} not found. Run generate_sandbox.py first."
        )

    # Aggregate detector counts → total per 15-min interval
    det = pd.read_csv(det_path, parse_dates=["timestamp"])
    agg = (
        det.groupby("timestamp")
        .agg(total_count=("vehicle_count", "sum"),
             rain=("rain", "first"),
             temp_c=("temp_c", "first"))
        .reset_index()
        .sort_values("timestamp")
    )

    # Signal phase → dominant phase active per 15-min window
    if sig_path.exists():
        sig = pd.read_csv(sig_path, parse_dates=["timestamp"])
        sig["window"] = sig["timestamp"].dt.floor("15min")
        # majority phase per window
        phase_mode = (
            sig[sig["signal_state"] == "GREEN_ON"]
            .groupby("window")["phase_number"]
            .agg(lambda s: s.mode()[0] if len(s) else 0)
            .reset_index()
            .rename(columns={"window": "timestamp", "phase_number": "dominant_phase"})
        )
        agg = agg.merge(phase_mode, on="timestamp", how="left")
        agg["dominant_phase"] = agg["dominant_phase"].fillna(0).astype(int)
    else:
        agg["dominant_phase"] = 0

    # Calendar features
    agg["hour"]        = agg["timestamp"].dt.hour
    agg["dow"]         = agg["timestamp"].dt.dayofweek
    agg["is_weekend"]  = (agg["dow"] >= 5).astype(float)
    agg["hour_sin"]    = np.sin(2 * math.pi * agg["hour"] / 24)
    agg["hour_cos"]    = np.cos(2 * math.pi * agg["hour"] / 24)
    agg["dow_sin"]     = np.sin(2 * math.pi * agg["dow"]  / 7)
    agg["dow_cos"]     = np.cos(2 * math.pi * agg["dow"]  / 7)
    agg["phase_sin"]   = np.sin(2 * math.pi * agg["dominant_phase"] / 4)
    agg["phase_cos"]   = np.cos(2 * math.pi * agg["dominant_phase"] / 4)

    feature_cols = [
        "total_count", "rain", "temp_c",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "phase_sin", "phase_cos",
    ]

    data = agg[feature_cols].values.astype(np.float32)

    # Normalise: z-score per column
    mean = data.mean(axis=0)
    std  = data.std(axis=0) + 1e-8
    data_norm = (data - mean) / std

    # Target: total_count at each horizon (un-normalised, rescaled back for loss)
    count_mean  = mean[0]
    count_std   = std[0]

    return data_norm, data[:, 0], mean, std, feature_cols, count_mean, count_std


def make_sequences(data_norm, targets, seq_len, horizons):
    max_h = max(horizons)
    xs, ys = [], []
    for i in range(len(data_norm) - seq_len - max_h):
        xs.append(data_norm[i : i + seq_len])
        ys.append([targets[i + seq_len - 1 + h] for h in horizons])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train(epochs: int = 40, hidden: int = 64, lr: float = 1e-3,
          batch: int = 64, val_split: float = 0.15):

    data_norm, raw_counts, mean, std, feature_cols, c_mean, c_std = load_and_prepare(DATA_DIR)
    X, y = make_sequences(data_norm, raw_counts, SEQ_LEN, HORIZONS)

    # Train / val split (chronological)
    n_val   = int(len(X) * val_split)
    X_tr, X_va = X[:-n_val], X[-n_val:]
    y_tr, y_va = y[:-n_val], y[-n_val:]

    tr_ds  = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va_ds  = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    tr_dl  = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    va_dl  = DataLoader(va_ds, batch_size=batch)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                           "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = TrafficLSTM(input_size=len(feature_cols), hidden=hidden).to(device)
    optim     = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_ds)

        # ── validate ──
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item() * len(xb)
        va_loss /= len(va_ds)

        scheduler.step(va_loss)

        if epoch % 5 == 0 or epoch == 1:
            rmse_15 = math.sqrt(va_loss)   # approximate; all horizons have similar scale
            print(f"Epoch {epoch:3d}/{epochs}  train={tr_loss:.2f}  val={va_loss:.2f}  "
                  f"RMSE≈{rmse_15:.1f} veh")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save ──
    torch.save({
        "model_state":   best_state,
        "input_size":    len(feature_cols),
        "hidden":        hidden,
        "feature_cols":  feature_cols,
        "horizons":      HORIZONS,
        "seq_len":       SEQ_LEN,
        "norm_mean":     mean.tolist(),
        "norm_std":      std.tolist(),
        "count_mean":    float(c_mean),
        "count_std":     float(c_std),
        "val_loss":      best_val_loss,
    }, MODEL_PATH)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Best val loss: {best_val_loss:.2f}  (RMSE ≈ {math.sqrt(best_val_loss):.1f} vehicles)")


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-horizon LSTM traffic forecasting model.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(epochs=args.epochs, hidden=args.hidden, lr=args.lr, batch=args.batch)
