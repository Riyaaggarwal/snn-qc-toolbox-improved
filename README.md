# SpikeSense Studio

Time-series classification with spiking neural network features and explainable model reports.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Riyaaggarwal/snn-qc-toolbox-improved/main/app.py)

<img width="1992" height="444" alt="SpikeSense Studio screenshot" src="https://github.com/user-attachments/assets/69f622ac-90b4-45a4-858e-72b0069de240" />

SpikeSense Studio extends the original SNN-QC toolbox into a general-purpose Streamlit product for
reusable time-series experiments. Original toolbox and research prototype by **Dr. Ravi Kumar Jha**,
Intelligent Systems Research Centre, Ulster University — **Jha-R@ulster.ac.uk**.

---

## Features

- Upload labelled time-series data or explore with the built-in EEG demo (60 trials, 128 timepoints, 14 channels).
- Try curated public examples for wearables, industrial process monitoring, ECG screening, and robot/IoT sensors.
- Preview data shape and class balance before analysis.
- Delta-encode signals, simulate a NeuCube reservoir, and extract spiking feature vectors.
- Compare Classical SVM, Logistic Regression, and an experimental Quantum Kernel SVM.
- Download feature tables, per-class metrics, confusion matrix, and experiment configs as CSV / JSON.
- Session-level experiment history with one-click CSV download.
- All defaults configurable via `config.yaml` — no code changes needed.

---

## Deploy on Streamlit Community Cloud (shareable link)

1. **Fork or push this repo** to your GitHub account (must be public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set the main file to `app.py` → click **Deploy**.
4. Streamlit installs `requirements.txt` automatically. First deploy takes ~3–5 minutes.
5. Share the generated `https://your-app-name.streamlit.app` link with anyone.

> **Tip:** Streamlit Community Cloud is free for public repos. No server setup needed.

---

## Local installation

```bash
git clone https://github.com/Riyaaggarwal/snn-qc-toolbox-improved.git
cd snn-qc-toolbox-improved
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t spikesense-studio .
docker run -p 8501:8501 spikesense-studio
```

---

## Configuration

Edit `config.yaml` to change defaults without touching Python:

```yaml
app:
  name: "SpikeSense Studio"          # browser tab title and hero header
  tagline: "..."                     # subtitle shown in the hero banner

defaults:
  spike_sensitivity: 0.8            # delta encoding threshold
  validation_folds: 5               # stratified k-fold CV
  reservoir_c: 0.4                  # reservoir connectivity probability
  reservoir_l: 0.169                # distance decay for connectivity
  random_seed: 42                   # reproducibility seed
  max_displayed_features: 14        # default features pre-selected in model report
```

---

## Built-in public examples

The **Product examples** source includes compact, balanced subsets from the
[UEA/UCR Time Series Classification Archive](https://www.timeseriesclassification.com/):

| Product area | Dataset | Shape in app | Why it is useful |
|--------------|---------|--------------|------------------|
| Wearable sensor data | BasicMotions | 24 samples × 100 timepoints × 6 axes | Smartwatch accelerometer and gyroscope movement classification |
| Industrial sensor monitoring | Wafer | 24 samples × 19 timepoints × 8 temporal segments | Normal vs faulty semiconductor wafer process traces |
| Healthcare signal screening | ECG200 | 24 samples × 16 timepoints × 6 temporal segments | Heartbeat morphology classification |
| IoT / robot sensing | SonyAIBORobotSurface1 | 18 samples × 14 timepoints × 5 temporal segments | Robot surface-state classification from embedded sensor traces |

Regenerate these files with:

```bash
python scripts/build_public_examples.py
```

---

## Data formats

### Single combined table

One row per sample-timepoint. Select columns in the UI after upload.

| sample_id | time | label   | sensor_1 | sensor_2 |
|-----------|------|---------|----------|----------|
| trial_001 | 0    | class_a | 0.12     | 0.32     |
| trial_001 | 1    | class_a | 0.18     | 0.29     |
| trial_002 | 0    | class_b | 0.76     | 0.15     |

### Multiple sample files

One CSV per sample (rows = timepoints, columns = features). Plus a separate labels CSV with one
label per file, matched in sorted filename order.

Download templates directly from the app's **CSV Templates** expander.

---

## Using the NeuCube library directly

```python
from neucube import Reservoir
from neucube.encoder import Delta
from neucube.sampler import SpikeCount
from neucube.training import STDP
from neucube.qfeatures import extract_features

# Encode raw time-series into spike trains
X = Delta(threshold=0.8).encode_dataset(data)   # (samples, timepoints, features)

# Create and simulate the reservoir
res = Reservoir(inputs=X.shape[2], c=0.4, l=0.169)
s_act = res.simulate(X, train=True, learning_rule=STDP(), mem_thr=0.01)

# Pool spike activity into feature vectors
state_vectors = SpikeCount().sample(s_act)        # (samples, neurons)
features = extract_features(state_vectors, res.w_in)  # (samples, channels)
```

---

## Acknowledgements

The EEG dataset and NeuCube software environment are made available by Auckland University of
Technology at [https://kedri.aut.ac.nz/neucube](https://kedri.aut.ac.nz/neucube).

## References

1. Jha, R. K., Kasabov, N., Bhattacharyya, S., Coyle, D., & Prasad, G. (2025). A hybrid spiking
   neural network-quantum framework for spatio-temporal data classification: a case study on EEG
   data. *EPJ Quantum Technology*, 12(1), 1–23.
   https://doi.org/10.1140/epjqt/s40507-025-00443-1

2. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping, learning and
   understanding of spatio-temporal brain data. *Neural Networks*, 52, 62–76.
   https://doi.org/10.1016/j.neunet.2014.01.006
