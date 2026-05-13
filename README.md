# Supernova Neutrino Detection with Deep Learning

A deep learning pipeline for identifying supernova neutrino signals in simulated Liquid-Argon Time-Projection Chamber (LArTPC) detector data. The project explores the feasibility of an early-warning system for supernova events by combining convolutional classification, energy regression, and image denoising on sparse, high-dimensional detector scans buried in noise.

## Motivation

When a star goes supernova, an enormous burst of neutrinos arrives at Earth hours before the visible light does. Detecting that burst in real time would give astronomers an early warning to point their telescopes at the right patch of sky. LArTPC detectors are well-suited to catching these neutrinos, but the signals are sparse and buried under electronic noise and ambient radioactive backgrounds. This project tackles three core problems on simulated detector data:

1. **Classification** — distinguishing real neutrino events from noise.
2. **Regression** — recovering particle energies from noisy images.
3. **Denoising** — reconstructing clean signals using a U-Net autoencoder.

## Dataset

The dataset consists of 10,000 simulated LArTPC events, each represented as a 100×100 pixel image of energy deposition over a slice of space and time. Each image is paired with a 64-dimensional metadata vector describing the underlying physics — including neutrino and electron energies, initial- and final-state particle counts, and PDG codes for the particles involved.

Two custom noise models were built on top of the clean signals:

- **Gaussian electronic noise** at varying standard deviations to simulate sensor interference.
- **2D Gaussian "blob" backgrounds** modeling radioactive decay (e.g., Argon-39 beta decay), with randomised position, amplitude (up to ~0.565 MeV), and spatial spread.

## Key Results

| Task | Model | Result |
|---|---|---|
| Clean vs. empty classification | AlexNet-style CNN | 1.0 accuracy (all noise levels) |
| Noisy classification (up to 13σ noise) | Custom Mini-ResNet-18 | 98% accuracy (0-4$$\sigma$$ noise levels) |
| Neutrino / electron energy regression | Tuned residual CNN | 8.0% MAE |
| Signal denoising | Residual U-Net | Recovered sparse tracks via noise-prediction (log-compressed inputs, residual learning) |

## Technical Implementations

### Classification
A lightweight AlexNet-inspired CNN establishes the baseline for clean vs. empty discrimination. A custom **Mini-ResNet-18** built with the Keras functional API handles the harder noisy case — using identity residual blocks with Batch Normalisation, `LeakyReLU` and `Swish` activations, and Global Average Pooling to extract sparse tracks from up to 13σ Gaussian noise combined with radioactive backgrounds.

### Energy Regression
The residual architecture was adapted into a continuous regression model with a linear output, predicting neutrino and electron energies in MeV. Hyperparameters (filter depth, activations, blocks per stack, L2 regularisation, dropout, learning rate) were tuned with **Keras Tuner** using both Random Search and Bayesian Optimisation strategies. The result was a 45% MAE reduction versus the baseline.

### U-Net Denoiser
A symmetric U-Net with skip connections recovers clean tracks from noisy detector images. The key design choice is residual learning: rather than predicting the sparse clean signal, the network predicts the noise field and subtracts it from the input. This avoids the zero-output collapse that direct-reconstruction targets are prone to under MAE loss — when most pixels in the target are zero, predicting all-zeros is a strong local minimum that a direct model can get stuck in. Inputs use `log1p` dynamic-range compression to keep faint tracks visible after normalisation, since a handful of high-intensity outlier pixels would otherwise dominate the scale and wash out the rest of the signal.

## Evaluation

- **Classification:** Precision, Recall, F1-Score, and Binary Crossentropy loss, swept across a sliding scale of noise intensities.
- **Regression:** MSE and MAE, plus Matplotlib dashboards showing predicted-vs-true scatter plots, residual distributions, and MAE binned by energy range.
- Noise overlays were re-randomised at each epoch to prevent the model from memorising specific noise realisations and to reduce overfitting.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras, Keras Tuner
- **ML Utilities:** Scikit-learn
- **Numerics:** NumPy
- **Visualisation:** Matplotlib

## Repository Structure

```
.
├── supernova_neutrino_detection.ipynb   # Full pipeline notebook
├── requirements.txt
└── README.md
```

The entire pipeline — data loading, noise simulation, model definitions, training, hyperparameter tuning, and evaluation — lives in a single Jupyter notebook for end-to-end reproducibility.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook supernova_neutrino_detection.ipynb
```

## Author

**[Your Name]** — [LinkedIn](#) · [Email](#) · [Portfolio](#)
