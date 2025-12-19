# preprocessing/resample.py
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def resample_signal(
    x: np.ndarray,
    fs_in: float,
    fs_out: float,
    axis: int = -1,
) -> np.ndarray:
    """
    Resample a signal using polyphase filtering (recommended for biosignals).

    Args:
        x: Signal array (..., T)
        fs_in: original sampling rate
        fs_out: target sampling rate
        axis: time axis

    Returns:
        Resampled signal with same shape except time dimension.
    """
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("fs_in and fs_out must be > 0")
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)

    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x_m = np.moveaxis(x, axis, -1)

    a = int(round(fs_out * 1000))
    b = int(round(fs_in * 1000))
    g = gcd(a, b)
    up = a // g
    down = b // g

    y = resample_poly(x_m, up=up, down=down, axis=-1).astype(np.float32)
    y = np.moveaxis(y, -1, axis)
    return y


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Resample a numpy .npy signal file using polyphase resampling.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy")
    parser.add_argument("--fs_in", type=float, required=True, help="Original sampling rate")
    parser.add_argument("--fs_out", type=float, required=True, help="Target sampling rate")
    parser.add_argument("--axis", type=int, default=-1, help="Time axis (default: -1)")
    args = parser.parse_args()

    inp = np.load(args.input)
    out = resample_signal(inp, fs_in=args.fs_in, fs_out=args.fs_out, axis=args.axis)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved resampled signal to: {args.output} | shape={out.shape}")
