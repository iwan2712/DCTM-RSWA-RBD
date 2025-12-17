# Dynamic Compression Transformer Multimodal (DCTM)

**Adaptive, efficient, and interpretable deep learning framework for automated REM Sleep Behavior Disorder (RBD) detection from multimodal PSG signals.**

---

## Overview
REM Sleep Behavior Disorder (RBD) is a clinically significant parasomnia and a prodromal marker of α-synucleinopathies. Manual diagnosis via polysomnography (PSG) is labor-intensive and subject to inter-rater variability.  
This repository presents **DCTM**, a **signal-aware Transformer-based framework** designed for **robust, accurate, and computationally efficient** detection of REM Sleep Without Atonia (RSWA) and RBD under real-world recording conditions.

---

## Key Features
- **SQI-guided Gating**  
  Dynamically down-weights or bypasses unreliable modalities (EEG, EOG, EMG, SpO₂) based on signal quality.
- **Cross-modal Transformer with Mixture-of-Experts (MoE)**  
  Enables adaptive multimodal specialization beyond static fusion.
- **Dynamic Compression**  
  Elastic depth/width adjustment, pruning, quantization, and knowledge distillation to reduce inference cost.
- **Progressive Inference**  
  Early-exit decisions with conditional modality activation (EOG+EMG → EEG when needed).
- **Interpretability**  
  Attention maps, modality-wise importance, and failure-case analysis.

---

## Model Pipeline

**Preprocessing → SQI Estimation → SQI-Guided Gating → Modality-specific Encoding → Cross-modal Transformer → Mixture-of-Experts Fusion → Dynamic Compression → Progressive Inference**
---

## Experimental Highlights
- **Datasets**: CAP RBD, Sleep-EDF Expanded, HMC Sleep Database  
- **Performance**:
  - F1-score: **0.947**
  - AUC: **0.95**
  - Cohen’s κ: **0.85**
- **Efficiency**:
  - Latency: **28 ms / epoch**
  - FLOPs: **3.2 GFLOPs**
  - Memory: **360 MB**
- **Robustness**:
  - Stable under noise, motion artifacts, and missing channels

---

## Reproducibility
- **Framework**: PyTorch 2.x
- **Training**: AdamW + cosine LR, mixed precision (FP16)
- **Evaluation**:
  - Subject- and epoch-level metrics
  - Ablation, robustness, and interpretability analyses
- **Validation**:
  - Subject-wise 5-fold CV
  - Leave-One-Dataset-Out (LODO)

