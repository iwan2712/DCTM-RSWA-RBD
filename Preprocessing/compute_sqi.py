# preprocessing/compute_sqi.py
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Integrate PSD in [fmin, fmax]. psd shape (..., F), freqs shape (F,)"""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.zeros(psd.shape[:-1], dtype=np.float32)
    return np.trapz(psd[..., mask], freqs[mask], axis=-1).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_sqi(
    x: np.ndarray,
    fs: float,
    modality: str = "eeg",
    mains_hz: float = 50.0,
    nperseg: int | None = None,
    axis: int = -1,
) -> np.ndarray:
    """
    Compute a simple Signal Quality Index (SQI) in [0, 1] for gating.

    Components:
      1) Band SNR proxy (target band vs noise band)
      2) Line-noise penalty near mains (50/60 Hz)
      3) Clipping penalty (flat-top saturation proxy)
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")

    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.moveaxis(x, axis, -1)
    T = x.shape[-1]
    if T < 16:
        return np.zeros(x.shape[:-1], dtype=np.float32)

    if nperseg is None:
        nperseg = int(min(max(int(4 * fs), 64), T))

    freqs, psd = welch(x, fs=fs, nperseg=nperseg, axis=-1, scaling="density")

    modality = modality.lower()
    if modality == "eeg":
        target = (0.5, 35.0)
        noise = (35.0, min(90.0, fs / 2 - 1e-6))
    elif modality == "eog":
        target = (0.1, 15.0)
        noise = (15.0, min(60.0, fs / 2 - 1e-6))
    elif modality == "emg":
        target = (10.0, min(100.0, fs / 2 - 1e-6))
        noise = (0.5, 10.0)
    else:
        raise ValueError("modality must be one of: 'eeg', 'eog', 'emg'")

    p_target = _bandpower(psd, freqs, *target)
    p_noise = _bandpower(psd, freqs, *noise) + 1e-8
    snr_proxy = p_target / p_noise

    line_bw = 1.0
    p_line = _bandpower(psd, freqs, mains_hz - line_bw, mains_hz + line_bw)
    p_total = _bandpower(psd, freqs, 0.5, min(90.0, fs / 2 - 1e-6)) + 1e-8
    line_ratio = p_line / p_total

    absx = np.abs(x)
    hi = np.percentile(absx, 99.5, axis=-1)
    near_hi = (absx >= (0.98 * hi[..., None])).mean(axis=-1)

    snr_score = _sigmoid((np.log10(snr_proxy + 1e-8) - 0.0) / 0.5)
    line_score = 1.0 - np.clip(line_ratio / 0.2, 0.0, 1.0)
    clip_score = 1.0 - np.clip(near_hi / 0.05, 0.0, 1.0)

    sqi = (0.55 * snr_score + 0.25 * line_score + 0.20 * clip_score).astype(np.float32)
    return np.clip(sqi, 0.0, 1.0)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Compute SQI (0-1) from a numpy .npy signal file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy (shape: [T] or [C,T])")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy (SQI per channel)")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    parser.add_argument("--modality", type=str, default="eeg", choices=["eeg", "eog", "emg"], help="Signal modality")
    parser.add_argument("--mains_hz", type=float, default=50.0, help="Line frequency (50 or 60)")
    parser.add_argument("--axis", type=int, default=-1, help="Time axis (default: -1)")
    parser.add_argument("--nperseg", type=int, default=0, help="Welch nperseg (0=auto)")

    args = parser.parse_args()
    inp = np.load(args.input)
    nperseg = None if args.nperseg == 0 else args.nperseg
    sqi = compute_sqi(inp, fs=args.fs, modality=args.modality, mains_hz=args.mains_hz, nperseg=nperseg, axis=args.axis)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, sqi)
    print(f"Saved SQI to: {args.output} | shape={sqi.shape} | min={float(np.min(sqi)):.3f} max={float(np.max(sqi)):.3f}")
