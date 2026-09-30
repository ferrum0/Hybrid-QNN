"""Training/evaluation loop, capacity matching, and the encoding/baseline
comparison harness (requirements 3-4 from the brief: hybrid backprop through
PyTorch with a quantum layer inside, and accuracy/time vs a classical
baseline of matched capacity).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RunResult:
    name: str
    train_acc_history: list = field(default_factory=list)
    val_acc_history: list = field(default_factory=list)
    test_acc: float = 0.0
    train_seconds: float = 0.0
    n_params: int = 0


def _accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def train_model(
    model: nn.Module,
    X_train, y_train, X_val, y_val, X_test, y_test,
    epochs: int = 30,
    lr: float = 0.1,
    batch_size: int | None = None,
    name: str = "model",
    verbose: bool = True,
) -> RunResult:
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = batch_size or len(Xtr)

    result = RunResult(name=name, n_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    start = time.perf_counter()
    n = len(Xtr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()

        train_acc = _accuracy(model, Xtr, ytr)
        val_acc = _accuracy(model, Xva, yva)
        result.train_acc_history.append(train_acc)
        result.val_acc_history.append(val_acc)
        if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
            print(f"  [{name}] epoch {epoch + 1:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    result.train_seconds = time.perf_counter() - start
    result.test_acc = _accuracy(model, Xte, yte)
    return result


def match_capacity_hidden_size(in_dim: int, n_classes: int, target_params: int) -> int:
    """Pick an MLP hidden width so Linear(in,h)+Linear(h,n_classes) has a
    parameter count close to `target_params` (the quantum model's count),
    for an apples-to-apples "same capacity" classical baseline."""
    best_h, best_diff = 1, float("inf")
    for h in range(1, 256):
        n_params = (in_dim + 1) * h + (h + 1) * n_classes
        diff = abs(n_params - target_params)
        if diff < best_diff:
            best_h, best_diff = h, diff
    return best_h


def print_comparison_table(results: list[RunResult]) -> None:
    header = f"{'model':<28}{'params':>8}{'train_s':>10}{'test_acc':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<28}{r.n_params:>8}{r.train_seconds:>10.2f}{r.test_acc:>10.3f}")
