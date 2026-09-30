"""Parametrized quantum circuits (PQCs) and their PyTorch bridge.

Two variational classifiers are implemented, sharing the same trainable
block (Rot + ring-of-CNOTs entangler) so accuracy/speed differences in
train.py's comparison are attributable to the *encoding*, not the ansatz:

- angle encoding with data re-uploading (Perez-Salinas et al. 2020): the
  classical features are re-embedded via RY rotations before every
  variational layer. Re-uploading is what gives a shallow, few-qubit circuit
  enough expressivity to act as a universal function approximator -- a
  single embedding + linear variational layer is provably too weak.
- amplitude encoding: the feature vector is embedded once as the state's
  amplitudes (exponential compression: n qubits <-> 2**n features), then the
  same variational block is applied. It is *not* re-uploaded, because
  AmplitudeEmbedding prepares an absolute state (it overwrites, rather than
  rotates, the register) -- re-invoking it mid-circuit would erase every
  variational step that came before it.

Both QNodes use diff_method="parameter-shift": an exact, hardware-compatible
gradient rule (not a simulator trick like backprop/adjoint) that evaluates
each parameter at theta +- pi/2 and takes a finite combination of the two
results. It is what a real QPU has to use, which is the whole point of
demonstrating it here.
"""
from __future__ import annotations

import math
from typing import Callable

import pennylane as qml
import torch


def _variational_block(weights_layer, n_qubits: int) -> None:
    for q in range(n_qubits):
        qml.Rot(*weights_layer[q], wires=q)
    for q in range(n_qubits - 1):
        qml.CNOT(wires=[q, q + 1])
    if n_qubits > 2:
        qml.CNOT(wires=[n_qubits - 1, 0])


def build_angle_reupload_qnode(n_qubits: int, n_layers: int, dev=None):
    """Angle encoding + data re-uploading variational classifier."""
    dev = dev or qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        for l in range(n_layers):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            _variational_block(weights[l], n_qubits)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return circuit, weight_shapes


def build_amplitude_qnode(n_qubits: int, n_layers: int, dev=None):
    """Amplitude encoding (single embedding) + variational classifier."""
    dev = dev or qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        for l in range(n_layers):
            _variational_block(weights[l], n_qubits)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return circuit, weight_shapes


BUILDERS: dict[str, Callable] = {
    "angle": build_angle_reupload_qnode,
    "amplitude": build_amplitude_qnode,
}


def small_angle_init(std: float = 0.1) -> Callable[[torch.Tensor], torch.Tensor]:
    """Near-identity initializer: mitigates barren plateaus by starting the
    circuit close to the identity (all rotations ~0) instead of drawing
    angles uniformly from [0, 2*pi]. See src/barren_plateaus.py for why this
    helps and where it stops helping."""
    def init(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init


def make_quantum_torch_layer(
    encoding: str,
    n_qubits: int,
    n_layers: int,
    diff_method: str = "parameter-shift",
    init: str = "uniform",
    init_std: float = 0.1,
) -> qml.qnn.TorchLayer:
    """Build a qml.qnn.TorchLayer for the given encoding, ready to drop into
    an nn.Module. `init="uniform"` reproduces PennyLane's default (U[0, 2*pi]
    per weight); `init="small"` uses the near-identity initializer instead.
    """
    if encoding not in BUILDERS:
        raise ValueError(f"Unknown encoding '{encoding}', expected one of {list(BUILDERS)}")

    dev = qml.device("default.qubit", wires=n_qubits)
    circuit, weight_shapes = BUILDERS[encoding](n_qubits, n_layers, dev=dev)
    if diff_method != "parameter-shift":
        circuit = qml.QNode(circuit.func, dev, interface="torch", diff_method=diff_method)

    init_method = None
    if init == "small":
        init_method = {"weights": small_angle_init(init_std)}
    elif init != "uniform":
        raise ValueError(f"Unknown init '{init}', expected 'uniform' or 'small'")

    return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)
