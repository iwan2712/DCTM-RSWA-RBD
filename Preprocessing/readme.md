

## Preprocessing Notes

All datasets are processed using **identical preprocessing pipelines** to ensure fair comparison:

- Resampling:
  - EEG/EOG/EMG → 200 Hz
  - SpO₂ preserved at 1 Hz
- Filtering:
  - EEG/EOG: 0.3–35 Hz
  - EMG: 10–100 Hz
- Epoching:
  - 30-second epochs
  - REM epochs identified via hypnogram annotations
- RSWA labeling:
  - SINBAR criteria (tonic and phasic activity)

Relevant scripts are provided in:

