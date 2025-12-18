# Dataset Information and Access Instructions

This repository does **NOT** distribute raw polysomnography (PSG) data due to data usage and ethical restrictions.  
All datasets used in this study are **publicly available** and must be downloaded directly from their original sources.

This directory provides the **expected folder structure**, dataset descriptions, and instructions for correct data preparation to ensure reproducibility of the reported results.

---

## 1. Datasets Used

### 1.1 CAP Sleep Database – RBD Subset (Primary Dataset)

- **Purpose**: Supervised training and evaluation for RSWA and RBD detection  
- **Subjects**: 60 (30 RBD, 30 controls)  
- **Signals**:
  - EEG (C3–A2, O2–A1)
  - EOG
  - EMG (mentalis, tibialis)
  - SpO₂
- **Annotations**:
  - RBD diagnosis
  - REM Sleep Without Atonia (RSWA), following SINBAR criteria
- **Sampling rate**:
  - EEG/EOG/EMG: 200 Hz
  - SpO₂: 1 Hz

📥 **Download**:  
https://physionet.org/content/capslpdb/

📄 **Reference**:  
Ferri et al., *Sleep*, 2023.

---

### 1.2 Sleep-EDF Expanded Dataset (Pre-training)

- **Purpose**: Self-supervised and supervised pre-training
- **Subjects**: 153 healthy adults
- **Signals**:
  - EEG (Fpz–Cz, Pz–Oz)
  - EOG
  - EMG
  - SpO₂
- **Annotations**:
  - Sleep stages (W, N1–N3, REM)
- **Sampling rate**:
  - EEG/EOG/EMG: 100 Hz
  - SpO₂: 1 Hz

📥 **Download**:  
https://physionet.org/content/sleep-edfx/

📄 **Reference**:  
Goldberger et al., *PhysioNet*.

---

### 1.3 HMC Sleep Database (Robustness & Generalization)

- **Purpose**: Robustness testing under clinical noise conditions
- **Subjects**: 208 clinical PSG recordings
- **Signals**:
  - EEG
  - EOG
  - EMG
  - SpO₂
- **Annotations**:
  - Sleep stages
  - Arousals
  - Desaturation events
- **Sampling rate**:
  - EEG/EOG/EMG: 200 Hz
  - SpO₂: 1 Hz

📥 **Download**:  
https://physionet.org/content/hmc-sleepdb/

---

## 2. Expected Directory Structure

After downloading and extracting the datasets, please organize them as follows:

