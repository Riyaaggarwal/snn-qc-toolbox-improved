import torch
from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, X_in):
        pass


class SpikeCount(Sampler):
    """Sum spikes per neuron across the time dimension → (batch, neurons)."""

    def __init__(self):
        super().__init__()

    def sample(self, spike_activity):
        return spike_activity.sum(axis=1)