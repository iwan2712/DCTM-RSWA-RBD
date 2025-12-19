# preprocessing/build_epochs_dataset.py
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from .resample import resample_signal
from .bandpass_filter import bandpass_filter
from .emg_rms import emg_rms_envelope
from .epoching import epoch_signal
from .compute_sqi import compute_sqi
from .spo2_processing import process_spo2


@dataclass
class BuildConfig:
    fs_target: float = 200.0
    epoch_sec: float = 30.0
    mains_hz: float = 50.0

    eeg_band: Tuple[float, float] = (0.3, 35.0)
    eog_band: Tuple[float, float] = (0.3, 35.0)
    emg_band: Tuple[float, float] = (10.0, 100.0)

    emg_rms_window_sec: float = 0.2
    emg_rms_smooth_hz: float = 5.0


def _load_npy(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(str(path))


def build_epochs_from_npy_folder(
    subject_dir: str | pathlib.Path,
    out_path: str | pathlib.Path,
    config: BuildConfig = BuildConfig(),
    fs_map: Optional[Dict[str, float]] = None,
    channels: Optional[List[str]] = None,
    save_sqi: bool = True,
) -> None:
    """
    Build a single .npz containing aligned epochs across modalities.

    Expected inputs inside subject_dir:
      - eeg.npy, eog.npy, emg.npy, spo2.npy  (arrays [..., T])
      - labels.npy (optional) epoch-level labels aligned to 30s epochs

    Output .npz contains:
      X_eeg, X_eog, X_emg, X_spo2  -> (E, C, L)  (SpO2 also (E,1,Ls))
      y (optional)
      sqi_eeg/eog/emg (optional) -> (E, C)
      meta_json -> JSON string with config + shapes

    Notes:
      - Assumes signals are time-aligned already.
      - For EDF parsing + hypnogram alignment, build dataset-specific loaders.
    """
    subject_dir = pathlib.Path(subject_dir)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if channels is None:
        channels = ["eeg", "eog", "emg", "spo2"]

    if fs_map is None:
        fs_map = {"eeg": 200.0, "eog": 200.0, "emg": 200.0, "spo2": 1.0}

    arrays = {ch: _load_npy(subject_dir / f"{ch}.npy") for ch in channels}

    # --- Resample & filter ---
    eeg = bandpass_filter(
        resample_signal(arrays["eeg"], fs_in=fs_map["eeg"], fs_out=config.fs_target, axis=-1),
        fs=config.fs_target,
        low=config.eeg_band[0],
        high=config.eeg_band[1],
        axis=-1,
    )
    eog = bandpass_filter(
        resample_signal(arrays["eog"], fs_in=fs_map["eog"], fs_out=config.fs_target, axis=-1),
        fs=config.fs_target,
        low=config.eog_band[0],
        high=config.eog_band[1],
        axis=-1,
    )
    emg = bandpass_filter(
        resample_signal(arrays["emg"], fs_in=fs_map["emg"], fs_out=config.fs_target, axis=-1),
        fs=config.fs_target,
        low=config.emg_band[0],
        high=config.emg_band[1],
        axis=-1,
    )
    emg_env = emg_rms_envelope(
        emg, fs=config.fs_target, window_sec=config.emg_rms_window_sec, smooth_lowpass_hz=config.emg_rms_smooth_hz, axis=-1
    )

    spo2 = process_spo2(arrays["spo2"], fs=fs_map["spo2"], axis=-1)

    # --- Epoching ---
    eeg_ep = epoch_signal(eeg, fs=config.fs_target, epoch_sec=config.epoch_sec, axis=-1, drop_last=True)
    eog_ep = epoch_signal(eog, fs=config.fs_target, epoch_sec=config.epoch_sec, axis=-1, drop_last=True)
    emg_ep = epoch_signal(emg_env, fs=config.fs_target, epoch_sec=config.epoch_sec, axis=-1, drop_last=True)
    spo2_ep = epoch_signal(spo2, fs=fs_map["spo2"], epoch_sec=config.epoch_sec, axis=-1, drop_last=True)

    def ensure_ECL(x_ep: np.ndarray) -> np.ndarray:
        if x_ep.ndim == 2:  # (E,L)
            return x_ep[:, None, :].astype(np.float32)
        if x_ep.ndim == 3:  # (C,E,L)
            return np.moveaxis(x_ep, 0, 1).astype(np.float32)
        raise ValueError(f"Unexpected epoch shape: {x_ep.shape}")

    X_eeg = ensure_ECL(eeg_ep)
    X_eog = ensure_ECL(eog_ep)
    X_emg = ensure_ECL(emg_ep)
    X_spo2 = ensure_ECL(spo2_ep)

    E = min(X_eeg.shape[0], X_eog.shape[0], X_emg.shape[0], X_spo2.shape[0])
    X_eeg, X_eog, X_emg, X_spo2 = X_eeg[:E], X_eog[:E], X_emg[:E], X_spo2[:E]

    out_dict = {"X_eeg": X_eeg, "X_eog": X_eog, "X_emg": X_emg, "X_spo2": X_spo2}

    labels_path = subject_dir / "labels.npy"
    if labels_path.exists():
        y = np.asarray(np.load(str(labels_path)))
        if y.shape[0] >= E:
            y = y[:E]
        else:
            y_pad = -1 * np.ones((E,), dtype=y.dtype)
            y_pad[: y.shape[0]] = y
            y = y_pad
        out_dict["y"] = y

    if save_sqi:
        def epoch_sqi(X: np.ndarray, modality: str) -> np.ndarray:
            E_, C_, L_ = X.shape
            flat = X.reshape(E_ * C_, L_)
            s = compute_sqi(flat, fs=config.fs_target, modality=modality, mains_hz=config.mains_hz, axis=-1)
            return s.reshape(E_, C_).astype(np.float32)

        out_dict["sqi_eeg"] = epoch_sqi(X_eeg, "eeg")
        out_dict["sqi_eog"] = epoch_sqi(X_eog, "eog")
        out_dict["sqi_emg"] = epoch_sqi(X_emg, "emg")

    meta = {
        "subject_dir": str(subject_dir),
        "config": config.__dict__,
        "fs_map": fs_map,
        "shapes": {k: list(v.shape) for k, v in out_dict.items() if isinstance(v, np.ndarray)},
    }
    out_dict["meta_json"] = np.array(json.dumps(meta), dtype=object)

    np.savez_compressed(str(out_path), **out_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build aligned epoch dataset (.npz) from subject folder with .npy signals.")
    parser.add_argument("--subject_dir", type=str, required=True, help="Folder containing eeg.npy/eog.npy/emg.npy/spo2.npy")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path")
    parser.add_argument("--fs_target", type=float, default=200.0, help="Target sampling rate for EEG/EOG/EMG (default: 200)")
    parser.add_argument("--epoch_sec", type=float, default=30.0, help="Epoch length seconds (default: 30)")
    parser.add_argument("--mains_hz", type=float, default=50.0, help="Mains frequency (50 or 60)")
    parser.add_argument("--fs_eeg", type=float, default=200.0)
    parser.add_argument("--fs_eog", type=float, default=200.0)
    parser.add_argument("--fs_emg", type=float, default=200.0)
    parser.add_argument("--fs_spo2", type=float, default=1.0)
    parser.add_argument("--no_sqi", action="store_true", help="Disable SQI computation")

    args = parser.parse_args()
    cfg = BuildConfig(fs_target=args.fs_target, epoch_sec=args.epoch_sec, mains_hz=args.mains_hz)
    fs_map = {"eeg": args.fs_eeg, "eog": args.fs_eog, "emg": args.fs_emg, "spo2": args.fs_spo2}
    build_epochs_from_npy_folder(args.subject_dir, args.out, config=cfg, fs_map=fs_map, save_sqi=not args.no_sqi)
    print(f"Saved dataset to: {args.out}")
