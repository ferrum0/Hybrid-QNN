"""Hybrid Quantum-Classical Neural Network classifier -- CLI entry point.

    python main.py train       --dataset iris --encoding angle
    python main.py compare     --dataset iris
    python main.py barren-plateaus
    python main.py qsvm        --dataset iris
    python main.py all

See README.md for the full write-up (architecture, encoding comparison,
parameter-shift rule, barren plateaus, quantum kernels).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src import data, qsvm, train
from src.barren_plateaus import gradient_variance_scan, plot_variance_scan
from src.models import ClassicalBaseline, HybridQNN


def cmd_train(args):
    split = data.load_dataset(args.dataset, n_classes=args.n_classes)
    Xtr, Xva, Xte, _ = data.prepare_features(split, args.n_qubits, args.encoding, args.encoder)
    model = HybridQNN(
        n_qubits=args.n_qubits, n_classes=split.n_classes, encoding=args.encoding,
        n_layers=args.n_layers, diff_method=args.diff_method, init=args.init,
    )
    print(f"HybridQNN[{args.encoding}] params={model.num_trainable_params()}")
    result = train.train_model(
        model, Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test,
        epochs=args.epochs, lr=args.lr, name=f"hybrid-{args.encoding}",
    )
    print(f"\nTest accuracy: {result.test_acc:.3f}  |  wall time: {result.train_seconds:.2f}s")


def cmd_compare(args):
    split = data.load_dataset(args.dataset, n_classes=args.n_classes)
    results = []

    for encoding in ("angle", "amplitude"):
        Xtr, Xva, Xte, _ = data.prepare_features(split, args.n_qubits, encoding, args.encoder)
        model = HybridQNN(
            n_qubits=args.n_qubits, n_classes=split.n_classes, encoding=encoding,
            n_layers=args.n_layers, diff_method=args.diff_method, init=args.init,
        )
        print(f"\n== HybridQNN [{encoding} encoding] ==")
        r = train.train_model(model, Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test,
                               epochs=args.epochs, lr=args.lr, name=f"hybrid-{encoding}")
        results.append(r)

    # Classical baseline of matched parameter count, trained on the same
    # (angle-encoded) preprocessed features so both branches see identical inputs.
    Xtr, Xva, Xte, _ = data.prepare_features(split, args.n_qubits, "angle", args.encoder)
    hidden = train.match_capacity_hidden_size(args.n_qubits, split.n_classes, results[0].n_params)
    baseline = ClassicalBaseline(args.n_qubits, hidden, split.n_classes)
    print(f"\n== Classical MLP baseline (hidden={hidden}, matched capacity) ==")
    r = train.train_model(baseline, Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test,
                           epochs=args.epochs, lr=args.lr, name="classical-mlp")
    results.append(r)

    print("\n" + "=" * 56)
    print("COMPARISON")
    print("=" * 56)
    train.print_comparison_table(results)


def cmd_barren_plateaus(args):
    print("Running gradient-variance scan (this evaluates many small circuits;"
          " a few seconds to ~1 minute depending on --max-qubits)...")
    scan = gradient_variance_scan(
        n_qubits_list=list(range(2, args.max_qubits + 1)),
        n_layers=args.n_layers, n_samples=args.n_samples,
    )
    for i, n in enumerate(scan["n_qubits"]):
        print(f"  n_qubits={n:2d}   Var[grad] uniform={scan['uniform'][i]:.3e}   "
              f"small-init={scan['small'][i]:.3e}")
    plot_variance_scan(scan, save_path="results/barren_plateaus.png")


def cmd_qsvm(args):
    split = data.load_dataset(args.dataset, n_classes=args.n_classes)
    Xtr, _, Xte, _ = data.prepare_features(split, args.n_qubits, "angle", args.encoder)
    # QSVM kernel matrices are O(n^2) circuit evaluations; subsample for a fast demo.
    n = min(args.n_samples, len(Xtr))
    idx = np.random.RandomState(0).choice(len(Xtr), n, replace=False)
    Xtr_s, ytr_s = Xtr[idx], split.y_train[idx]

    print(f"Fitting QSVM on {n} training points, {args.n_qubits} qubits...")
    _, qtrain_acc, qtest_acc = qsvm.fit_qsvm(Xtr_s, ytr_s, Xte, split.y_test, args.n_qubits)
    _, ctrain_acc, ctest_acc = qsvm.fit_classical_svm_baseline(Xtr_s, ytr_s, Xte, split.y_test)

    print(f"\n{'model':<24}{'train_acc':>12}{'test_acc':>12}")
    print("-" * 48)
    print(f"{'quantum kernel SVM':<24}{qtrain_acc:>12.3f}{qtest_acc:>12.3f}")
    print(f"{'classical RBF SVM':<24}{ctrain_acc:>12.3f}{ctest_acc:>12.3f}")


def cmd_all(args):
    print("### 1) Encoding + classical baseline comparison ###")
    cmd_compare(args)
    print("\n### 2) Barren plateaus ###")
    cmd_barren_plateaus(args)
    print("\n### 3) Quantum kernel method (QSVM) extension ###")
    cmd_qsvm(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=["iris", "digits"], default="iris")
    common.add_argument("--n-classes", type=int, default=None, help="restrict to first N classes")
    common.add_argument("--n-qubits", type=int, default=4)
    common.add_argument("--encoder", choices=["pca", "autoencoder"], default="pca")

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--encoding", choices=["angle", "amplitude"], default="angle")
    p_train.add_argument("--n-layers", type=int, default=3)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--lr", type=float, default=0.1)
    p_train.add_argument("--diff-method", default="parameter-shift")
    p_train.add_argument("--init", choices=["uniform", "small"], default="uniform")
    p_train.set_defaults(func=cmd_train)

    p_compare = sub.add_parser("compare", parents=[common])
    p_compare.add_argument("--n-layers", type=int, default=3)
    p_compare.add_argument("--epochs", type=int, default=30)
    p_compare.add_argument("--lr", type=float, default=0.1)
    p_compare.add_argument("--diff-method", default="parameter-shift")
    p_compare.add_argument("--init", choices=["uniform", "small"], default="uniform")
    p_compare.set_defaults(func=cmd_compare)

    p_bp = sub.add_parser("barren-plateaus")
    p_bp.add_argument("--max-qubits", type=int, default=8)
    p_bp.add_argument("--n-layers", type=int, default=5)
    p_bp.add_argument("--n-samples", type=int, default=60)
    p_bp.set_defaults(func=cmd_barren_plateaus)

    p_qsvm = sub.add_parser("qsvm", parents=[common])
    p_qsvm.add_argument("--n-samples", type=int, default=40)
    p_qsvm.set_defaults(func=cmd_qsvm)

    p_all = sub.add_parser("all", parents=[common])
    p_all.add_argument("--n-layers", type=int, default=3)
    p_all.add_argument("--epochs", type=int, default=30)
    p_all.add_argument("--lr", type=float, default=0.1)
    p_all.add_argument("--diff-method", default="parameter-shift")
    p_all.add_argument("--init", choices=["uniform", "small"], default="uniform")
    p_all.add_argument("--max-qubits", type=int, default=8)
    p_all.add_argument("--n-samples", type=int, default=40)
    p_all.set_defaults(func=cmd_all)

    return parser


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    args = build_parser().parse_args()
    args.func(args)
