# preprocessing/emg_rms.py
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _lowpass(x: np.ndarray, fs: float, cutoff: float = 5.0, order: int = 4, axis: int = -1) -> np.ndarray:
    nyq = fs / 2.0
    if not (0 < cutoff < nyq):
        raise ValueError(f"cutoff must be between 0 and fs/2. Got cutoff={cutoff}, fs={fs}")
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, x, axis=axis).astype(np.float32)


def emg_rms_envelope(
    emg: np.ndarray,
    fs: float,
    window_sec: float = 0.2,
    smooth_lowpass_hz: float = 5.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Compute EMG RMS envelope (useful for tonic/phasic activity characterization).

    Steps:
      1) sliding window RMS
      2) optional low-pass smoothing

    Args:
        emg: EMG signal array (..., T)
        fs: Sampling rate (Hz)
        window_sec: RMS window length in seconds (default 0.2s)
        smooth_lowpass_hz: Low-pass cutoff for smoothing envelope (default 5 Hz). Set None to disable.
        axis: Time axis index

    Returns:
        RMS envelope array with same shape as emg.
    """
    emg = np.asarray(emg, dtype=np.float32)
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")

    win = int(round(window_sec * fs))
    win = max(1, win)

    x = np.moveaxis(emg, axis, -1)
    T = x.shape[-1]

    pad = win // 2
    x_pad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")

    sq = x_pad ** 2
    csum = np.cumsum(sq, axis=-1)
    win_sum = csum[..., win:] - csum[..., :-win]
    rms = np.sqrt(win_sum / float(win)).astype(np.float32)

    if rms.shape[-1] > T:
        rms = rms[..., :T]
    elif rms.shape[-1] < T:
        rms = np.pad(rms, [(0, 0)] * (rms.ndim - 1) + [(0, T - rms.shape[-1])], mode="edge")

    if smooth_lowpass_hz is not None:
        rms = _lowpass(rms, fs=fs, cutoff=float(smooth_lowpass_hz), order=4, axis=-1)

    rms = np.moveaxis(rms, -1, axis)
    return rms


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Compute EMG RMS envelope from a numpy .npy EMG file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy (shape: [T] or [C,T])")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npy (RMS envelope)")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate (Hz)")
    parser.add_argument("--window_sec", type=float, default=0.2, help="RMS window seconds (default: 0.2)")
    parser.add_argument("--smooth_hz", type=float, default=5.0, help="Low-pass smoothing cutoff Hz (default: 5). Use 0 to disable.")
    parser.add_argument("--axis", type=int, default=-1, help="Time axis (default: -1)")

    args = parser.parse_args()
    inp = np.load(args.input)
    smooth = None if args.smooth_hz == 0 else args.smooth_hz
    out = emg_rms_envelope(inp, fs=args.fs, window_sec=args.window_sec, smooth_lowpass_hz=smooth, axis=args.axis)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, out)
    print(f"Saved RMS envelope to: {args.output}")
