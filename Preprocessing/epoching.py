# preprocessing/epoching.py
from __future__ import annotations

import numpy as np


def epoch_signal(
    x: np.ndarray,
    fs: float,
    epoch_sec: float = 30.0,
    axis: int = -1,
    drop_last: bool = True,
) -> np.ndarray:
    """
    Split a continuous signal into fixed-length epochs.

    Args:
        x: Signal array (..., T) where T is time samples (or time axis chosen by `axis`).
        fs: Sampling rate (Hz).
        epoch_sec: Epoch length in seconds (default 30s).
        axis: Time axis index (default -1).
        drop_last: If True, drop trailing samples that don't fill an epoch.

    Returns:
        Epochs with shape (..., E, L) where:
          E = number of epochs
          L = samples per epoch
        If input is (C, T) -> output (C, E, L)
        If input is (T,) -> output (E, L)
    """
    x = np.asarray(x)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if epoch_sec <= 0:
        raise ValueError("epoch_sec must be > 0")

    L = int(round(fs * epoch_sec))
    if L <= 0:
        raise ValueError("epoch length in samples must be > 0")

    x_moved = np.moveaxis(x, axis, -1)
    T = x_moved.shape[-1]

    if drop_last:
        E = T // L
        T_use = E * L
    else:
        E = int(np.ceil(T / L))
        T_use = E * L

    if T_use == 0:
        raise ValueError("Signal too short to form even one epoch.")

    if not drop_last and T_use > T:
        pad = T_use - T
        x_moved = np.pad(x_moved, [(0, 0)] * (x_moved.ndim - 1) + [(0, pad)], mode="constant")

    x_cut = x_moved[..., :T_use]
    new_shape = x_cut.shape[:-1] + (E, L)
    epochs = x_cut.reshape(new_shape)
    return epochs


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Epoch a numpy .npy signal file into fixed windows.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy (shape: [T] or [C,T])")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy (epochs)")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    parser.add_argument("--epoch_sec", type=float, default=30.0, help="Epoch length seconds (default: 30)")
    parser.add_argument("--axis", type=int, default=-1, help="Time axis (default: -1)")
    parser.add_argument("--keep_last", action="store_true", help="If set, pad and keep last partial epoch")

    args = parser.parse_args()
    inp = np.load(args.input)
    out = epoch_signal(inp, fs=args.fs, epoch_sec=args.epoch_sec, axis=args.axis, drop_last=not args.keep_last)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved epochs to: {args.output} | shape={out.shape}")
