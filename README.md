# CEG3004 DSP Mini-Project: Environmental Sound Classification
**Group:** Pr_42

---

## Overview

This project implements a robust audio classification pipeline for Environmental Sound Classification (ESC-50). The goal is to classify 5-second mono audio clips into 50 sound classes, with the pipeline designed to perform well under clean, noisy, and band-limited conditions.

---

## Repository Structure

```
CEG3004-Project/
├── README.md
├── requirements.txt
├── ceg3004_project_colab.py
└── results/
    ├── Pr_42_predictions.csv
    └── Pr_42_model.joblib
```

---

## How to Run

1. Open `ceg3004_project_colab.py` in Google Colab
2. Set `GROUP_ID = "Pr_42"` at the top (already set)
3. Run all cells in order
4. The model (`Pr_42_model.joblib`) and predictions (`Pr_42_predictions.csv`) will be auto-downloaded

### Requirements

Install dependencies by running Cell 1 in the notebook, or manually:

```bash
pip install numpy scipy pandas scikit-learn librosa soundfile tqdm
```

---

## DSP Pipeline

### 1. Preprocessing (`preprocess_audio`)

| Step | What it does | Why |
|---|---|---|
| Silence trimming | Removes leading/trailing silence using `top_db=30` | Reduces irrelevant frames that add noise to features |
| Fixed-length padding/truncation | Pads or truncates to exactly 5 seconds | Ensures consistent feature vector size across all clips |
| Pre-emphasis filter | Applies `y[n] - 0.97 * y[n-1]` | Boosts high-frequency content that is often attenuated, especially in band-limited clips |
| RMS normalisation | Scales waveform so RMS = 1 | Makes features invariant to recording loudness differences |

### 2. Feature Extraction (`extract_features`)

| Feature | Details | Why |
|---|---|---|
| MFCC + CMVN | 40 coefficients, `fmax=4000Hz`, cepstral mean-variance normalisation | CMVN removes channel/gain effects; `fmax=4000` keeps features in the frequency range preserved even under band-limiting |
| Delta + Delta-Delta | First and second order temporal derivatives of MFCC | Captures how features change over time, adding dynamic information |
| Log-Mel spectrogram | 128 mel bands, `fmin=50`, `fmax=4000Hz` | Captures spectral shape in a perceptually meaningful way |
| Spectral centroid | Mean + std over time | Describes the "brightness" of the sound |
| Spectral bandwidth | Mean + std over time | Describes how spread out the energy is around the centroid |
| Spectral rolloff | Mean + std over time | Indicates frequency below which most energy is concentrated |
| Spectral flatness | Mean + std over time | Distinguishes tonal sounds from noise-like sounds |
| Zero crossing rate | Mean + std over time | Indicates the noisiness/percussiveness of a signal |
| RMS energy | Mean + std over time | Captures loudness contour over time |

**Pooling strategy:** For MFCC and log-mel features, we use robust pooling — mean, std, median, 10th percentile, and 90th percentile — rather than just mean and std. This makes the representation more robust to short noise bursts.

**Robustness design rationale:** Setting `fmax=4000Hz` is a deliberate choice to ensure features are computed within the frequency range that survives band-limiting distortion. Pre-emphasis and CMVN together address gain and spectral tilt variations introduced by additive noise.

---

## Model

**Classifier:** Support Vector Machine with RBF kernel (`sklearn.svm.SVC`)

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True))
])
```

**Design choices:**
- `StandardScaler` normalises the feature vector before classification, which is required for SVM to work correctly
- `class_weight='balanced'` handles any class imbalance in the training set
- RBF kernel allows the model to learn non-linear decision boundaries, which are needed for 50-class audio classification

---

## Experiments

| Experiment | Classifier | Features | Val Macro-F1 |
|---|---|---|---|
| Baseline | Logistic Regression | 20 MFCCs + deltas (mean, std) | 0.487 |
| Final | SVM (RBF, C=10) | CMVN MFCCs + log-mel + spectral features, robust pooling | 0.531 |

The improved pipeline achieves a **+4.4% absolute improvement** in Macro-F1 on the validation set over the baseline. The main contributors are the richer feature set (log-mel + spectral features) and the stronger SVM classifier.

---

## Robustness Strategy

The submission set contains three versions of each clip: clean, noisy, and band-limited. The pipeline addresses each:

- **Noisy clips:** RMS normalisation and CMVN reduce sensitivity to additive noise by normalising gain and spectral tilt
- **Band-limited clips:** `fmax=4000Hz` on all frequency-domain features ensures the model relies only on the lower frequency range that survives bandwidth restriction; pre-emphasis recovers some of the high-frequency boost lost during this process

---

## Dataset

- Source: ESC-50 (Environmental Sound Classification)
- 2,000 clips, 50 classes, 40 clips per class
- Each clip: 5 seconds, mono, 16kHz
- Train/submission split provided by course staff
