"""Extension: quantum kernel methods (QSVM) as an alternative to the
variational hybrid classifier.

Instead of training circuit parameters, a fixed feature-map circuit
U(x)|0> embeds each classical point x into a quantum state, and the kernel
between two points is the squared overlap (fidelity) of their embedded
states, k(x, x') = |<0|U(x')^dagger U(x)|0>|^2 -- estimated here by running
U(x)^dagger U(x') and reading off the probability of measuring all zeros.
That kernel matrix is precomputed classically and handed to an ordinary
sklearn SVC(kernel="precomputed"); all the "quantum" part does is define a
similarity measure that would be classically hard to compute for feature
maps built from circuits that are hard to simulate. This sidesteps
barren-plateau training issues entirely (there is nothing to train on the
quantum side) at the cost of O(n^2) circuit evaluations for an n-point
kernel matrix.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from sklearn.svm import SVC


def make_quantum_kernel(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def overlap(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2) -> float:
        return float(overlap(x1, x2)[0])  # P(all-zeros) = |<phi(x1)|phi(x2)>|^2

    return kernel


def kernel_matrix(A: np.ndarray, B: np.ndarray, kernel) -> np.ndarray:
    return np.array([[kernel(a, b) for b in B] for a in A])


def fit_qsvm(X_train, y_train, X_test, y_test, n_qubits: int):
    kernel = make_quantum_kernel(n_qubits)
    K_train = kernel_matrix(X_train, X_train, kernel)
    K_test = kernel_matrix(X_test, X_train, kernel)

    clf = SVC(kernel="precomputed").fit(K_train, y_train)
    train_acc = clf.score(K_train, y_train)
    test_acc = clf.score(K_test, y_test)
    return clf, train_acc, test_acc


def fit_classical_svm_baseline(X_train, y_train, X_test, y_test):
    clf = SVC(kernel="rbf").fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    return clf, train_acc, test_acc
