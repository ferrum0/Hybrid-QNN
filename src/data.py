"""Dataset loading and classical dimensionality-reduction ("encoder") utilities.

The classical encoder's job is to compress raw features down to a size the
quantum circuit can consume:
  - angle encoding needs exactly `n_qubits` features (one rotation per qubit)
  - amplitude encoding needs at most `2**n_qubits` features (one qubit encodes
    two amplitudes, so n qubits span a 2**n-dimensional state vector)

Two encoders are provided: PCA (fast, deterministic, linear) and a tiny
autoencoder (nonlinear, learned) as the prompt asked for both options.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Split:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    n_classes: int
    raw_dim: int


def load_dataset(name: str, n_classes: int | None = None, seed: int = 0) -> Split:
    """Load a small tabular dataset.

    name: "iris" (4 features, 3 classes) or "digits" (8x8=64 features, 10
    classes) -- `digits` stands in for "MNIST in reduced dimensionality"
    since it needs no network download and PCA/autoencoder compression onto
    a handful of qubits behaves the same way real MNIST would.
    """
    if name == "iris":
        ds = load_iris()
        X, y = ds.data, ds.target
    elif name == "digits":
        ds = load_digits()
        X, y = ds.data, ds.target
    else:
        raise ValueError(f"Unknown dataset '{name}', expected 'iris' or 'digits'")

    if n_classes is not None:
        mask = y < n_classes
        X, y = X[mask], y[mask]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return Split(
        X_train, X_val, X_test, y_train, y_val, y_test,
        n_classes=len(np.unique(y)), raw_dim=X.shape[1],
    )


class PCAEncoder:
    """Linear classical encoder: standardize + PCA down to `n_components`."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.pca: PCA | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_
        n_components = min(self.n_components, Xs.shape[1])
        self.pca = PCA(n_components=n_components, random_state=0).fit(Xs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        out = self.pca.transform(Xs)
        if out.shape[1] < self.n_components:
            pad = np.zeros((out.shape[0], self.n_components - out.shape[1]))
            out = np.hstack([out, pad])
        return out


class AutoEncoder(nn.Module):
    """Small nonlinear encoder: bottleneck autoencoder trained with MSE.

    Offered as the "autoencoder" alternative to PCA mentioned in the brief;
    PCA is the default because it is deterministic and needs no training,
    which keeps the quantum-vs-classical comparison from being confounded by
    encoder-training noise.
    """

    def __init__(self, in_dim: int, bottleneck: int, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def fit_transform(self, X: np.ndarray, epochs: int = 200, lr: float = 1e-2) -> np.ndarray:
        Xt = torch.tensor(X, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            recon = self(Xt)
            loss = loss_fn(recon, Xt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return self.encoder(Xt).numpy()

    def transform(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.encoder(torch.tensor(X, dtype=torch.float32)).numpy()


def scale_for_angle_encoding(X_train, X_val, X_test, feature_range=(-np.pi, np.pi)):
    """Angle-embedding rotations are 2*pi periodic; map features into
    [-pi, pi] (fit on train only) so rotation angles stay well separated
    instead of wrapping around and colliding."""
    scaler = MinMaxScaler(feature_range=feature_range).fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def prepare_features(split: Split, n_qubits: int, encoding: str, encoder: str = "pca"):
    """Build encoder-compressed, encoding-appropriate feature arrays.

    Returns (X_train, X_val, X_test) as float32 numpy arrays ready to feed
    into the quantum layer, plus the fitted encoder (for inspection/reuse).
    """
    if encoding == "angle":
        target_dim = n_qubits
    elif encoding == "amplitude":
        target_dim = min(split.raw_dim, 2 ** n_qubits)
    else:
        raise ValueError(f"Unknown encoding '{encoding}'")

    if target_dim >= split.raw_dim:
        Xtr, Xva, Xte = split.X_train, split.X_val, split.X_test
        enc = None
    elif encoder == "pca":
        enc = PCAEncoder(target_dim).fit(split.X_train)
        Xtr, Xva, Xte = enc.transform(split.X_train), enc.transform(split.X_val), enc.transform(split.X_test)
    elif encoder == "autoencoder":
        enc = AutoEncoder(split.raw_dim, target_dim)
        Xtr = enc.fit_transform(split.X_train)
        Xva, Xte = enc.transform(split.X_val), enc.transform(split.X_test)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'")

    if encoding == "angle":
        Xtr, Xva, Xte = scale_for_angle_encoding(Xtr, Xva, Xte)
    # amplitude encoding only needs L2-normalizable vectors; AmplitudeEmbedding
    # in the quantum layer normalizes and zero-pads automatically.

    to_f32 = lambda a: np.asarray(a, dtype=np.float32)
    return to_f32(Xtr), to_f32(Xva), to_f32(Xte), enc
