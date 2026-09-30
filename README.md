# Hybrid Quantum-Classical Neural Network (Hybrid QNN)

A small, runnable research-style project pairing a classical encoder with a
variational quantum circuit for classification, built with **PennyLane +
PyTorch**. It targets the classic "hybrid QNN" recipe used in QML papers:

```
raw features --[PCA / autoencoder]--> compressed features
             --[angle or amplitude embedding]--> quantum state
             --[parametrized quantum circuit, data re-uploading]--> qubit expectation values
             --[classical linear readout]--> class logits
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pins autoray==0.6.11; pennylane 0.38 breaks with newer autoray
python main.py all                   # runs everything below, ~1-2 minutes
```

Individual pieces:

```bash
python main.py train   --dataset iris --encoding angle       # single model
python main.py compare --dataset iris                        # angle vs amplitude vs classical baseline
python main.py barren-plateaus                                # gradient-variance scan + plot
python main.py qsvm    --dataset iris                          # quantum kernel SVM extension
```

`--dataset digits` swaps in sklearn's 8x8-digit dataset (64 raw features) as
a stand-in for "MNIST in reduced dimensionality" — it needs no network
download, and PCA/autoencoder compression onto a handful of qubits behaves
identically to how it would on real MNIST, just faster to iterate on.

## Architecture

- **Classical encoder** (`src/data.py`): `PCAEncoder` standardizes then
  projects onto `n_components` principal axes; `AutoEncoder` is a tiny
  bottleneck MLP trained with MSE as the nonlinear alternative the brief
  asked about. Angle encoding needs exactly `n_qubits` compressed features;
  amplitude encoding needs at most `2**n_qubits` (since n qubits span a
  `2**n`-dimensional state vector) — for Iris (4 raw features) that's already
  under the `2**4 = 16` budget for 4 qubits, so no compression happens at all
  in that case; for `digits` (64 features) both encoders actually compress.

- **Quantum layer** (`src/quantum_layer.py`): both encodings share the same
  trainable block — a `Rot` gate per qubit plus a ring of `CNOT`s — so that
  accuracy/speed differences are attributable to the *encoding*, not the
  ansatz. Wrapped as `qml.qnn.TorchLayer`, which makes the weights ordinary
  `nn.Parameter`s that live inside a PyTorch `nn.Module` (requirement 3:
  hybrid backprop through PyTorch with a quantum layer embedded in it).

- **Classical readout** (`src/models.py`): a plain `nn.Linear(n_qubits,
  n_classes)` on top of the per-qubit `<Z>` expectation values.

## (1) Angle vs. amplitude encoding

| | Angle encoding | Amplitude encoding |
|---|---|---|
| Qubits needed for `d` features | `d` (1 feature/qubit) | `ceil(log2(d))` (exponential compression) |
| Re-uploadable mid-circuit? | Yes — `RY(x)` rotates the *current* state | No — `AmplitudeEmbedding` **overwrites** the register to a specific absolute state, so calling it again mid-circuit erases every variational step before it |
| Circuit depth to prepare | O(1) rotations | O(2^n) gates in general (state preparation is exponentially expensive to implement exactly) |
| Used here with | data re-uploading (see below) | single embedding + variational block only |

**Data re-uploading** (Pérez-Salinas et al., 2020): a single embedding
followed by one variational layer is a fairly weak function class — it's
essentially only able to fit within the classical hull it started in. By
*re-embedding the classical features before every variational layer*, the
circuit becomes able to approximate arbitrary functions of the input to
higher precision as more layers/re-uploads are added, without adding more
qubits, which is exactly why it's the go-to trick for extracting more
expressivity out of few-qubit devices. It only makes sense for embeddings
that *rotate* the state (angle, basis) rather than embeddings that *set* it
outright (amplitude) — that's why the amplitude branch here uses a single
embedding.

Empirically (`python main.py compare --dataset iris --epochs 30`, 4 qubits,
3 layers, 51 trainable parameters in each branch):

```
model                         params   train_s  test_acc
--------------------------------------------------------
hybrid-angle                      51      ~13      ~0.87-0.97
hybrid-amplitude                  51       ~8      ~0.87-0.97
classical-mlp                     51    ~0.02      ~0.90-0.97
```

(Exact numbers vary run to run — Iris is small and easy, so all three reach
similar accuracy; the qualitative and reproducible finding is the **training
time** column, discussed next.)

## (2)+(3) Parameter-shift rule and hybrid backprop

Both QNodes are built with `diff_method="parameter-shift"`: an exact
analytic gradient rule that works by evaluating the *same* circuit at
`theta + pi/2` and `theta - pi/2` and combining the two results —

```
dC/dtheta = [C(theta + pi/2) - C(theta - pi/2)] / 2
```

for any gate generated by an operator with eigenvalues +-1 (true for the
`Rot`/`RY`/`RZ` rotations used here). This is the gradient rule real quantum
hardware has to use (no access to intermediate statevectors the way
classical backprop or the simulator-only `adjoint`/`backprop` methods have),
which is why it's used by default here even though it's slower — it's what
you'd need to keep if you swapped `default.qubit` for real hardware.
`qml.qnn.TorchLayer` wires this gradient into PyTorch's autograd graph, so
`loss.backward()` transparently backprops through the classical readout,
*through* the quantum circuit via parameter-shift, into the classical
encoder upstream — one optimizer, one `loss.backward()` call, no manual
gradient plumbing.

**Cost**: parameter-shift needs 2 circuit evaluations per parameter per
sample (both forward pass and backward pass evaluate the circuit — a
36-parameter circuit costs ~73 simulator calls per sample). That's the
mechanical reason the comparison table above shows the quantum branches
training 2-3 orders of magnitude slower than the classical MLP for the
*same number of trainable parameters* on a simulator — a cost that would
look different again on real hardware (shot noise, queueing) but the
"quantum parameters are simply more expensive to differentiate" point
holds either way. Pass `--diff-method backprop` to `train`/`compare` to see
the same circuits train fast under a simulator-only shortcut gradient — useful
for quick iteration, but it wouldn't run on a real QPU.

## (4) Barren plateaus

See `src/barren_plateaus.py` for the full writeup in the module docstring;
summary:

**The problem** (McClean et al., 2018): for a sufficiently expressive/deep
random parametrized circuit, `Var[dC/dtheta]` shrinks *exponentially* with
the number of qubits, `O(1/2^n)`. The gradient isn't just small near some
minimum — it's small **everywhere**, so gradient descent gets no directional
signal at all, and the problem gets *worse* as you scale qubits up, exactly
when you'd want quantum models to start pulling ahead of classical ones.
Mechanically, this happens because sufficiently expressive/deep circuits
built from generic gates rapidly approach a Haar-random unitary (a
"2-design"), and the variance of any observable's expectation value under
Haar-random unitaries shrinks exponentially in the number of qubits.

**Mitigation demonstrated here** (Grant et al., 2019): initialize rotation
angles from a narrow `N(0, 0.1)` instead of the "natural-looking" uniform
`U[0, 2*pi]`. A near-zero-angle circuit is close to the identity, which is
a specific, non-random operation — the circuit starts *outside* the
Haar-random-concentration regime, where gradients are not exponentially
suppressed, giving the optimizer real signal to work with early in training.

Run `python main.py barren-plateaus` to reproduce: it scans qubit counts,
measures `Var[d<Z0>/dtheta]` over many random draws under both
initializations, and saves `results/barren_plateaus.png` — the uniform-init
curve should fall off much faster than the small-angle curve as qubits
increase.

**Honest caveat** (also noted in the module docstring): small-angle init
counters plateaus caused by *starting inside* the concentrated regime, not
plateaus caused by circuit *depth* — with enough re-uploading layers, even a
near-identity-initialized circuit eventually drifts back toward Haar-random
behavior over the course of training. It's one mitigation among several
(others: local rather than global cost functions, layer-wise/greedy
training, and problem-informed rather than generic ansatze) and this repo
implements only the initialization one, since that's what the brief asked
for.

## Extension: quantum kernel methods (QSVM)

`src/qsvm.py` swaps the trainable variational classifier for a **fixed**
feature-map circuit whose only job is to define a similarity measure: embed
each point `x` via `U(x)|0>`, then define

```
k(x, x') = |<0| U(x)^dagger U(x') |0>|^2
```

estimated by running `U(x)^dagger U(x')` and reading off the probability of
measuring all-zeros. That kernel matrix is precomputed and handed to an
ordinary `sklearn.svm.SVC(kernel="precomputed")`. Since nothing on the
quantum side is trained, this sidesteps barren plateaus entirely — the
trade-off is `O(n^2)` circuit evaluations to build an `n`-point kernel
matrix, so it doesn't scale to large datasets the way the variational
classifier does. Run `python main.py qsvm` to compare it against a classical
RBF-kernel SVM baseline on the same data.

## Project layout

```
main.py                    CLI entry point (train / compare / barren-plateaus / qsvm / all)
src/data.py                 dataset loading, PCA + autoencoder classical encoders
src/quantum_layer.py        angle-reupload and amplitude PQCs, TorchLayer bridge, inits
src/models.py                HybridQNN, ClassicalBaseline
src/train.py                 training loop, capacity matching, comparison table
src/barren_plateaus.py      gradient-variance experiment + plot
src/qsvm.py                  quantum kernel + SVC
results/                    generated plots land here
```

## Caveats / what this is not

- All circuits run on `default.qubit` (exact statevector simulation) — there
  is no shot noise or hardware error model here.
- Datasets are intentionally small (Iris, 8x8 digits) so a laptop-scale
  parameter-shift training run finishes in seconds to a couple minutes; this
  is a from-scratch educational/research demo, not a benchmark suite.
- "Matched capacity" for the classical baseline means "closest achievable
  parameter count for an `in -> hidden -> out` MLP," not a claim of
  equivalent representational power — parameter count is a common but
  imperfect capacity proxy for very different function classes.
