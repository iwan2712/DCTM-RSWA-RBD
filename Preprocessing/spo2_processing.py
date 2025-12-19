# preprocessing/spo2_processing.py
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt


def process_spo2(
    spo2: np.ndarray,
    fs: float = 1.0,
    median_kernel_sec: int = 5,
    valid_range: tuple[float, float] = (50.0, 100.0),
    max_step_per_sec: float = 8.0,
    axis: int = -1,
) -> np.ndarray:
    """
    SpO2 preprocessing:
      - Replace out-of-range values with NaN then interpolate
      - Median filter to suppress motion spikes
      - Step-change limiter (artifact handling)

    Args:
        spo2: array (..., T)
        fs: sampling rate in Hz (often 1 Hz)
        median_kernel_sec: median filter kernel in seconds (odd kernel enforced)
        valid_range: plausible SpO2 range
        max_step_per_sec: maximum plausible change per second (%/sec)
        axis: time axis

    Returns:
        cleaned spo2 array (..., T), float32
    """
    spo2 = np.asarray(spo2, dtype=np.float32)
    spo2 = np.moveaxis(spo2, axis, -1)
    T = spo2.shape[-1]

    lo, hi = valid_range
    x = spo2.copy()
    bad = (x < lo) | (x > hi) | ~np.isfinite(x)
    x[bad] = np.nan

    # interpolate NaNs along time
    if np.isnan(x).any():
        t = np.arange(T)
        x_flat = x.reshape(-1, T)
        for i in range(x_flat.shape[0]):
            xi = x_flat[i]
            nans = np.isnan(xi)
            if nans.all():
                xi[:] = 95.0
                continue
            if nans.any():
                xi[nans] = np.interp(t[nans], t[~nans], xi[~nans])
        x = x_flat.reshape(x.shape)

    # median filter
    k = int(round(median_kernel_sec * fs))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    if k > 1:
        x = medfilt(x, kernel_size=[1] * (x.ndim - 1) + [k]).astype(np.float32)

    # step-change limiter
    max_step = float(max_step_per_sec) / float(fs)
    dx = np.diff(x, axis=-1)
    dx = np.clip(dx, -max_step, max_step)
    x_limited = np.concatenate([x[..., :1], x[..., :1] + np.cumsum(dx, axis=-1)], axis=-1).astype(np.float32)

    x_limited = np.moveaxis(x_limited, -1, axis)
    return x_limited


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="SpO2 median filtering + artifact handling for numpy .npy SpO2 file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy")
    parser.add_argument("--fs", type=float, default=1.0, help="Sampling rate (default: 1.0)")
    parser.add_argument("--median_kernel_sec", type=int, default=5, help="Median kernel seconds (default: 5)")
    parser.add_argument("--axis", type=int, default=-1, help="Time axis (default: -1)")
    args = parser.parse_args()

    inp = np.load(args.input)
    out = process_spo2(inp, fs=args.fs, median_kernel_sec=args.median_kernel_sec, axis=args.axis)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved processed SpO2 to: {args.output} | shape={out.shape}")
