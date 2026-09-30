"""Hybrid quantum-classical model and a matched-capacity classical baseline."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.quantum_layer import make_quantum_torch_layer


class HybridQNN(nn.Module):
    """classical encoder (done upstream in data.py, e.g. PCA) -> PQC -> classical readout.

    The PQC's per-qubit <Z> expectation values (in [-1, 1]) are the "quantum
    features" fed to a small linear readout layer that maps them to class
    logits -- this readout is the "classical readout layer" from the brief.
    """

    def __init__(self, n_qubits: int, n_classes: int, encoding: str, n_layers: int,
                 diff_method: str = "parameter-shift", init: str = "uniform", init_std: float = 0.1):
        super().__init__()
        self.q_layer = make_quantum_torch_layer(
            encoding, n_qubits, n_layers, diff_method=diff_method, init=init, init_std=init_std
        )
        self.readout = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.q_layer(x))

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassicalBaseline(nn.Module):
    """Plain MLP baseline. `hidden` is chosen by the caller (see
    train.match_capacity_hidden_size) so its parameter count is close to the
    HybridQNN it's being compared against -- an "equal capacity" classical
    competitor rather than an arbitrarily sized one.
    """

    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
