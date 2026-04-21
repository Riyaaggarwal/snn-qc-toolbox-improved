# SNN-QC-Toolbox
SNN-QC: Spiking Neural Network-Quantum Computational Toolbox

<img width="1992" height="444" alt="image" src="https://github.com/user-attachments/assets/69f622ac-90b4-45a4-858e-72b0069de240" />


# Outline

## Key Features

- Implementation of Neucube in Python
- Written with PyTorch for faster computations
- Ability to capture and process patterns in spatio-temporal data
- Spatio-temporal trained spiking feature extraction
- Integration of NeuCube with advanced Quantum Kernel
- Generic Streamlit workbench for built-in or uploaded time-series datasets
- Classical and quantum classifier options for extracted SNN features

## Demostrations

- neucube_demo.py: existing NeuCube architecture
- neucube_improved_demo.py: an improved feature selection approach with the existing NeuCube architecture
- snn-qc_demo.py: an integrated advanced snn-quantum architecture 

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

The app now supports three dataset sources:

- **Built-in EEG demo**: uses the bundled wrist-movement EEG sample data.
- **Single combined CSV**: use one row per sample-timepoint. Select the sample ID column, label column, optional time/order column, and numeric feature columns in the UI.
- **Multiple sample CSVs**: upload one CSV per sample, where every file has the same `timepoints x features` shape, plus a labels CSV with one label per sample file in sorted filename order.

After loading a dataset, the app applies Delta spike encoding, runs NeuCube spatio-temporal learning, extracts SNN features, and supports:

- Quantum Kernel SVM for two selected classes and two selected features
- Classical SVM for binary or multiclass classification
- Logistic Regression for binary or multiclass classification

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
