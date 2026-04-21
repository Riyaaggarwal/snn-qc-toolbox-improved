# SpikeSense Studio
Time-series classification with spiking features and explainable model reports.

SpikeSense Studio extends the original SNN-QC toolbox into a more general-purpose Streamlit product experience.
Original toolbox and research prototype developed by **Dr. Ravi Kumar Jha**, Intelligent Systems Research
Centre, Ulster University. Original app contact: **Jha-R@ulster.ac.uk**.

<img width="1992" height="444" alt="image" src="https://github.com/user-attachments/assets/69f622ac-90b4-45a4-858e-72b0069de240" />


## Key Features

- Upload labelled time-series data or use the included EEG example.
- Preview data shape and class balance before analysis.
- Generate spiking feature tables for downstream modelling.
- Compare classical SVM, logistic regression, and an advanced quantum-kernel SVM mode.
- Export feature tables, classification metrics, and experiment configuration.
- Preserve credit and references for the original SNN-QC research toolbox.

## Installation

Clone this repository and install the Python dependencies:

```bash
git clone git@github.com:Riyaaggarwal/snn-qc-toolbox-improved.git
cd snn-qc-toolbox-improved
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

The app supports three dataset sources:

- **Example dataset**: uses the bundled wrist-movement EEG sample data.
- **Single uploaded table**: use one row per sample-timepoint. Select the sample ID column, label column, optional time/order column, and numeric feature columns in the UI.
- **Multiple sample files**: upload one CSV per sample, where every file has the same `timepoints x features` shape, plus a labels CSV with one label per sample file in sorted filename order.

After loading a dataset, the app prepares the signals, runs the analysis engine, extracts spiking features, and supports:

- Classical SVM for binary or multiclass classification
- Logistic Regression for binary or multiclass classification
- Quantum Kernel SVM for two selected classes and two selected features

The core functionality of NeuCube-Py revolves around the `reservoir` class, which represents the spiking neural network model. Here is a basic example of how to use NeuCube-Py:

```python
from neucube import Reservoir
from neucube.encoder import Delta
from neucube.sampler import SpikeCount

# Create a Reservoir 
res = Reservoir(inputs=14)

# Convert data to spikes
X = Delta().encode_dataset(data)

# Simulate the Reservior
out = res.simulate(X)

# Extract state vectors from the spiking activity
state_vec = SpikeCount.sample(out)

# Perform prediction and validation
# ...

```
## Acknowledgments
The EEG dataset and NeuCube software environment are kindly made available from the Auckland University of Technology at: [https://kedri.aut.ac.nz/neucube]. Users of the SNN-QC toolbox should cite the publications listed in the References section.

# References
[1]. Jha, R. K., Kasabov, N., Bhattacharyya, S., Coyle, D., & Prasad, G. (2025). A hybrid spiking neural network-quantum framework for spatio-temporal data classification: a case study on EEG data. EPJ Quantum Technology, 12(1), 1-23. https://doi.org/10.1140/epjqt/s40507-025-00443-1

[2]. Kasabov, N. (2014). NeuCube: A spiking neural network architecture for mapping, learning and understanding of spatio-temporal brain data. Neural networks, 52, 62-76. https://doi.org/10.1016/j.neunet.2014.01.006
