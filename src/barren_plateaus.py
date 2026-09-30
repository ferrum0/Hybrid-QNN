"""Barren plateaus: empirical demonstration + mitigation via initialization.

The barren plateau phenomenon (McClean et al., 2018, "Barren plateaus in
quantum neural network training landscapes"): for a sufficiently expressive
/ deep random parametrized circuit, the variance of the cost gradient with
respect to any parameter shrinks *exponentially* with the number of qubits.
Concretely, Var[dC/dtheta] ~ O(1/2^n). Two consequences that make this a
genuine training problem rather than a minor slowdown:

1. Because the gradient concentrates around zero everywhere (not just near
   a minimum), gradient-based optimizers see a landscape that is flat almost
   everywhere -- there's no "downhill" signal to follow, so more shots or a
   fancier optimizer don't help without a change to the circuit/init/cost.
2. It gets *worse* as you add qubits, exactly when you'd want quantum
   circuits to start outperforming classical models.

Why it happens: as the circuit approaches a Haar-random unitary (a 2-design)
-- which happens quickly for deep circuits built from generic gates -- the
expectation value of any observable, averaged over random parameters, has a
variance that shrinks exponentially in n. Random uniform initialization
(theta ~ U[0, 2*pi]) is close to worst-case here, because it starts the
optimizer already inside the plateau.

Mitigation demonstrated here: near-identity / small-angle initialization
(Grant et al., 2019, "An initialization strategy for addressing barren
plateaus in parametrized quantum circuits"). Initializing rotation angles
from a narrow N(0, sigma) instead of U[0, 2*pi] keeps each variational block
close to the identity, so the circuit starts in a region correlated with the
identity operation, where gradients are provably not exponentially
suppressed. Important caveat this experiment is honest about: this helps
against the *initialization* being stuck on a plateau, not against plateaus
induced by circuit *depth* -- as n_layers grows, even small-angle-initialized
circuits eventually drift back toward Haar-random behavior. Other mitigations
not implemented here: local (per-qubit) instead of global cost functions,
layer-wise/greedy training, and problem-informed ansatze.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def _random_layer_circuit(n_qubits: int, n_layers: int, dev):
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(weights[l, q, 0], wires=q)
                qml.RZ(weights[l, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.expval(qml.PauliZ(0))

    return circuit


def gradient_variance_scan(
    n_qubits_list: list[int],
    n_layers: int = 5,
    n_samples: int = 60,
    seed: int = 0,
) -> dict:
    """For each n_qubits, draw `n_samples` random weight settings under both
    'uniform' (U[0, 2*pi]) and 'small' (N(0, 0.1)) initializations, compute
    d<Z0>/d(first RY angle) via the parameter-shift rule, and return the
    variance of that gradient across samples for each (init, n_qubits) pair.
    A shrinking 'uniform' curve and a comparatively flat 'small' curve
    reproduce the McClean et al. result and the Grant et al. mitigation.
    """
    rng = np.random.default_rng(seed)
    results = {"uniform": [], "small": []}

    for n_qubits in n_qubits_list:
        dev = qml.device("default.qubit", wires=n_qubits)
        circuit = _random_layer_circuit(n_qubits, n_layers, dev)
        grad_fn = qml.grad(circuit)

        for init_name, sampler in (
            ("uniform", lambda: rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2))),
            ("small", lambda: rng.normal(0, 0.1, size=(n_layers, n_qubits, 2))),
        ):
            grads = []
            for _ in range(n_samples):
                weights = qml.numpy.array(sampler(), requires_grad=True)
                g = grad_fn(weights)
                grads.append(g[0, 0, 0])  # d<Z0>/d(theta for first RY)
            results[init_name].append(float(np.var(grads)))

    return {"n_qubits": list(n_qubits_list), **results}


def plot_variance_scan(scan: dict, save_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(scan["n_qubits"], scan["uniform"], "o-", label="uniform init  U[0, 2π]  (barren plateau)")
    ax.semilogy(scan["n_qubits"], scan["small"], "s-", label="small-angle init  N(0, 0.1)  (mitigated)")
    ax.set_xlabel("number of qubits")
    ax.set_ylabel("Var[∂⟨Z₀⟩/∂θ]  (log scale)")
    ax.set_title("Barren plateaus: gradient variance vs circuit width")
    ax.set_xticks(scan["n_qubits"])
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    return fig
