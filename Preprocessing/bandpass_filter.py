# preprocessing/bandpass_filter.py
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Args:
        x: Signal array (..., T) or (C, T) etc.
        fs: Sampling rate (Hz).
        low: Low cutoff (Hz).
        high: High cutoff (Hz).
        order: Butterworth order.
        axis: Time axis index.

    Returns:
        Filtered signal with same shape as x.
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if not (0 < low < high < fs / 2):
        raise ValueError(f"Cutoffs must satisfy 0 < low < high < fs/2. Got low={low}, high={high}, fs={fs}")

    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    x = np.asarray(x, dtype=np.float32)

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    y = filtfilt(b, a, x, axis=axis).astype(np.float32)
    return y


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Bandpass filter a numpy .npy signal file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy (shape: [T] or [C,T])")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    parser.add_argument("--low", type=float, required=True, help="Low cutoff (Hz)")
    parser.add_argument("--high", type=float, required=True, help="High cutoff (Hz)")
    parser.add_argument("--order", type=int, default=4, help="Butterworth order (default: 4)")
    parser.add_argument("--axis", type=int, default=-1, help="Time axis (default: -1)")

    args = parser.parse_args()
    inp = np.load(args.input)
    out = bandpass_filter(inp, fs=args.fs, low=args.low, high=args.high, order=args.order, axis=args.axis)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved filtered signal to: {args.output}")
