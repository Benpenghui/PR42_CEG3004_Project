# CEG3004 DSP Mini-Project: Environmental Sound Classification
**Group:** Pr_42

## How to Run
1. Open `ceg3004_project_colab.py` in Google Colab
2. Ensure `GROUP_ID = "Pr_42"` is set at the top
3. Run all cells in order
4. `Pr_42_model.joblib` and `Pr_42_predictions.csv` will be auto-downloaded

## Requirements
```bash
pip install numpy scipy pandas scikit-learn librosa soundfile tqdm
```

## Changes Made

### Preprocessing
| Step | Rationale |
|---|---|
| Silence trimming | Removes irrelevant frames at the start/end that would distort feature statistics |
| Fixed 5s length | Ensures consistent feature vector size across all clips |
| Pre-emphasis filter | Boosts high-frequency content that is attenuated in band-limited audio |
| RMS normalisation | Makes features invariant to recording loudness differences |

### Feature Extraction
| Feature | Rationale |
|---|---|
| CMVN MFCCs (40 coeffs, fmax=4000Hz) | CMVN removes channel/gain effects; fmax=4000Hz keeps features within frequency range preserved under band-limiting |
| Delta + Delta-Delta | Captures how features change over time, adding dynamic information |
| Log-mel spectrogram (fmin=50, fmax=4000Hz) | Captures spectral shape in a perceptually meaningful way |
| Spectral features (centroid, bandwidth, rolloff, flatness, ZCR, RMS) | Captures timbral and energy characteristics not covered by MFCCs alone |
| Robust pooling (mean, std, median, 10th/90th percentile) | More robust to short noise bursts than mean/std alone |

### Classifier
Three classifiers were evaluated. The best performing model was automatically selected and saved.

## Robustness Strategy
The submission set contains three versions of each clip: clean, noisy, and band-limited.

- **Noisy clips:** RMS normalisation and CMVN reduce sensitivity to additive noise by normalising gain and spectral tilt across frames
- **Band-limited clips:** Setting fmax=4000Hz on all frequency-domain features ensures the model relies only on the lower frequency range that survives bandwidth restriction; pre-emphasis compensates for high-frequency roll-off

## Results

| Experiment | Classifier | Macro-F1 |
|---|---|---|
| Baseline | Logistic Regression | 0.488 |
| Experiment 1 | SVM (RBF) | 0.532 |
| Final (best) | Random Forest | 0.564 |

Random Forest achieved the highest validation Macro-F1 and was selected as the final model.

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