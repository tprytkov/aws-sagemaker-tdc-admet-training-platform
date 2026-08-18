# GMC-MPNN BBB Phase-1 implementation plan

## Scope and decision status

This is an inspection and implementation plan, not an implementation. It is subordinate to
`docs/gmc_mpnn_bbb_phase1.md`: the target is binary `BBB_Martins`, class `1` means BBB-permeable,
all development decisions use train/validation only, and the previously evaluated BBB test set is
not loaded or used.

The upstream repository, paper, and checked-in code are not perfectly consistent. Phase 1 should
first reproduce the behavior of the exact inspected source revision with golden-reference tests.
Any deliberate correction or paper-aligned variant must be named and documented separately; it
must not silently replace the source-faithful baseline.

## Upstream provenance

| Item | Inspected value |
|---|---|
| Repository | [`MathIntelligence/GMC-MPNN-BBBP`](https://github.com/MathIntelligence/GMC-MPNN-BBBP) |
| Default branch | `main` |
| Exact commit | [`ae080431950832be43e51f7cd9b0f7d4203e2267`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/tree/ae080431950832be43e51f7cd9b0f7d4203e2267) |
| Inspection date | 2026-08-18 |
| Paper | Nguyen et al., “Geometric multi-color message passing graph neural networks for blood-brain barrier permeability prediction,” *Molecular Systems Design & Engineering* 11 (2026), 436–446, [DOI 10.1039/D5ME00175G](https://doi.org/10.1039/D5ME00175G) |
| Preprint inspected | [arXiv:2507.18926v5](https://arxiv.org/abs/2507.18926v5), 2025-12-05 |
| Repository license | **No root `LICENSE`, `LICENCE`, `COPYING`, `NOTICE`, or licensing statement was present.** |

The paper/preprint license is not a license for the source repository. Public visibility on GitHub
does not grant permission to copy, modify, or redistribute the GMC-MPNN code. Before any upstream
GMC-MPNN code is copied or adapted, obtain an explicit code license or author permission, or obtain
approval for an independently written implementation based on the published method. This is a
licensing gate, not legal advice.

The bundled `chemprop/` subtree has its own MIT `LICENSE.txt` and identifies itself as Chemprop
2.1.0. A file-level comparison against official Chemprop tag `v2.1.0` at commit
`bf00b3907121ec4ac982d48efb991f2646729b90` found that the executable library is essentially a
locally modified/pruned Chemprop 2.1.0 tree:

- the only substantive Python-library difference found is six added compatibility/warning lines
  in `chemprop/data/splitting.py` concerning `num_folds` and the changed return type;
- one CLI tutorial has a `.csv` to `.json` documentation correction;
- notebooks, examples, test data, and other non-runtime material are pruned; and
- no GMC-specific changes were made to message passing, featurization, datasets, normalization,
  loss, metrics, or optimization.

GMC-MPNN uses Chemprop's existing `V_f` extra-atom-feature interface. The geometric code lives
outside the Chemprop subtree. The bundled directory is therefore a lightly modified/pruned fork,
not an unmodified release, but its modifications do not appear necessary for GGL features.

## Upstream file map

| Path | Role | BBB classification requirement |
|---|---|---|
| `README.md` | Installation outline, feature command, multi-seed command, and external data/features link | Reference only; incomplete as a reproducible specification |
| `train_bbbp.py` | MoleculeNet BBBP data loading, internal split, feature scaling, exact BBBP model, training, validation, and test | Primary architecture/training reference; its split/test behavior must not be reused |
| `train_b3db_cls.py` | B3DB binary-classification model with different hyperparameters | Comparison reference only |
| `train_b3db_regression.py` | B3DB continuous `logBB` model and metrics | Unrelated to the Phase-1 classification task |
| `train.py` | Runs a selected training script over seeds and averages columns whose names begin with `test_` | Not reusable for our leakage-controlled workflow |
| `test.py` | Hard-codes one “best” kernel per dataset/seed, then retrains and evaluates the internally created test split | Upstream reproduction reference only; prohibited for our development protocol |
| `train.sh` | SLURM array example for seeds 0–4 | Operational example only; paths and script selection are placeholders |
| `utils/get_ggl_ligand_features.py` | Maps CSV `id` rows to MOL2 files, chooses a kernel, emits `.npz` arrays | Required behavior to reproduce |
| `utils/ggl_ligand.py` | Parses MOL2 coordinates/types and calculates six per-atom kernel summaries | Required behavior to reproduce |
| `utils/kernels.csv` | Grid of 1,600 exponential/Lorentz kernel settings | Required provenance for kernel selection |
| `utils/ligand_SYBYL_atom_types.csv` | 45 supported SYBYL atom types and radii | Required provenance for feature generation |
| `utils/extract_ggl_features.sh` | Site-specific SLURM job that generates all 1,600 kernels for B3DB classification | Reference only; contains upstream private-cluster paths and is not portable |
| `chemprop/` | Pruned/lightly modified Chemprop 2.1.0 | Use as a behavior reference; do not vendor the tree |

No conformer-generation, MOL2-conversion, OMEGA, RDKit fallback, OpenBabel fallback, atom-order
mapping, or molecular-cleaning script exists at the inspected revision. Full repository history also
shows only the same GGL extraction files, not a conformer generator. The external data link in the
README is not a substitute for a versioned preprocessing specification.

Pinned inspection links: [`README.md`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/README.md),
[`train_bbbp.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/train_bbbp.py),
[`train_b3db_cls.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/train_b3db_cls.py),
[`test.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/test.py),
[`utils/ggl_ligand.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/utils/ggl_ligand.py), and
[`utils/get_ggl_ligand_features.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/utils/get_ggl_ligand_features.py).

## What the checked-in model actually consumes

### Input and alignment contract

Feature extraction expects a CSV with `id` and `smiles`. `smiles` is read but not used by the
extractor. For every `id`, the extractor requires `<data_folder>/<id>.mol2`. BioPandas reads the
MOL2 atom table; each row supplies a SYBYL `atom_type` and Cartesian `x`, `y`, and `z` coordinate.

Training separately expects a CSV with `smiles` and, by default, `labels`. RDKit parses each SMILES
with `keep_h=False, add_h=False` to construct the ordinary molecular graph. A `.npz` file supplies
the geometric atom matrix for each molecule.

The `.npz` is created with `numpy.savez(output_path, *atom_features_list)`. It contains arrays named
`arr_0`, `arr_1`, and so on. Array `arr_i` is expected to correspond to CSV row `i`; row `j` within
that array is expected to correspond to RDKit heavy atom `j`. No molecule ID, SMILES, atom identity,
kernel metadata, atom count, checksum, or ordering metadata is stored in the archive, and training
uses `zip`, which can silently truncate unequal top-level lengths. The source checks per-molecule
atom counts only later when Chemprop creates a graph. It does not prove atom identity or ordering.

Our adapter must add a sidecar manifest and strict checks while retaining compatible numeric
arrays: molecule ID, canonical SMILES and hash, split, array key, MOL2 hash, kernel/cutoff, heavy-atom
count, ordered atom identities/types, coordinate-generation provenance, and feature hash. It must
fail on a missing/extra molecule, mismatched count, or unproved atom mapping.

### Three distinct feature classes

1. **Ordinary RDKit/Chemprop atom and bond features.** The checked-in code uses Chemprop 2.1.0's
   default v2 atom featurizer: 72 atom features (atomic-number vocabulary, degree, formal charge,
   chirality, hydrogen count, hybridization, aromaticity, and scaled mass) and 14 bond features
   (null flag, bond type, conjugation, ring membership, and stereochemistry).
2. **GMC-MPNN/GGL geometric atom features.** The checked-in extractor emits six columns per heavy
   atom: minimum, maximum, sum, mean, median, and standard deviation of retained kernel weights.
   These are passed as Chemprop `V_f` and concatenated to the 72 ordinary atom features before
   message passing. The checked-in model therefore receives 78 atom features and 14 bond features.
3. **Global molecular descriptors.** None are loaded. `x_d`/`X_d` is not supplied, and no global
   RDKit descriptor vector is concatenated after graph aggregation. Phase 1 must preserve this.

### GGL calculation in the checked-in source

Hydrogen rows whose MOL2 `atom_type` equals exactly `H` are removed. All remaining atoms are paired
with all remaining atoms. The code supports 45 SYBYL type strings and looks up a radius for each.
Unsupported type pairs receive `NaN` and are omitted by the later `NaN`-aware summaries.

For atoms `i,j`, distance `d_ij`, radii sum `R_ij = r_i + r_j`, scale `tau`, and power `kappa`, the
two implemented kernels are:

```text
exponential: exp(-(d_ij / (tau * R_ij)) ** kappa)
lorentz:     1 / (1 + (d_ij / (tau * R_ij)) ** kappa)
```

Self-interactions are set to `NaN`; distances greater than the cutoff are set to `NaN`. The shipped
cutoff and examples use 12.0 Å. `kernels.csv` contains 1,600 combinations: two kernel types,
`tau = 0.5, 1.0, ..., 10.0`, and `kappa = 0.5, 1.0, ..., 20.0`.

There is an apparent source defect or at least an undocumented discrepancy in covalent filtering:
the mask compares every distance in row `i` with the **row sum** of all pairwise radii for atom `i`,
not with the pair's radius sum. For ordinary molecules that threshold is very large, so the mask
usually removes nothing. Phase-1 source-faithful golden tests must capture this exact behavior.
Correcting it would be a separately named preprocessing variant.

The checked-in extractor does not explicitly build a separate array for each colored/atom-pair
subgraph. Atom types influence each pair's radius, but the six statistics summarize the full
retained row. This differs from the paper's description of atom-type-specific weighted colored
subgraphs.

### Paper/code discrepancies to resolve explicitly

| Topic | Paper/preprint v5 | Checked-in commit |
|---|---|---|
| Colored feature structure | Describes 12 corresponding atom-type subgraphs per atom | One all-heavy-atom kernel matrix; six summaries total per atom |
| Statistics | Text names sum, mean, median, min, max, std; displayed equation omits median | Exactly min, max, sum, mean, median, std |
| Ordinary atom encoding | Describes a 100-element atomic-number encoding consistent with Chemprop v1-style features | Uses Chemprop 2.1 default v2 encoding, 72 ordinary atom dimensions |
| Molecular aggregation | Method equation describes sum pooling | BBBP code uses normalized sum with divisor 57; B3DB classification uses mean pooling |
| Covalent exclusion | Paper refers to a covalent-bond exclusion rule | Source uses the row-summed-radius expression described above |
| 3D generation | Paper gives a high-level OMEGA-first/fallback workflow | No generation code, versions, seeds, or settings are present |

Until authors' artifacts or clarification resolve these differences, the exact checked-in behavior
is the code-reproduction target. A paper-faithful reconstruction must not be conflated with it.

## Exact upstream classification architectures

Both classifiers are one-task directed-bond MPNNs implemented with Chemprop 2.1.0
`nn.BondMessagePassing`. Unspecified message-passing defaults are hidden size 300, ReLU activation,
zero dropout, no bias in message matrices, directed messages, and 14-dimensional bond features.
The six normalized GGL columns are appended to ordinary atom features before the first message.

| Setting | MoleculeNet BBBP (`train_bbbp.py`) | B3DB classification (`train_b3db_cls.py`) |
|---|---:|---:|
| Atom input dimension | 78 = 72 ordinary + 6 GGL | 78 = 72 ordinary + 6 GGL |
| Bond input dimension | 14 | 14 |
| Message-passing hidden dimension | 300 (default) | 300 (default) |
| Message-passing depth | 5 | 3 |
| Message-passing activation/dropout | ReLU / 0 | ReLU / 0 |
| Aggregation | `NormAggregation(norm=57.0)` | `MeanAggregation()`; supplied `norm=4.0` is ignored |
| FFN task/output count | 1 binary logit | 1 binary logit |
| FFN hidden dimension | 900 | 700 |
| FFN hidden-layer count | `n_layers=2` | `n_layers=2` |
| FFN activation | LeakyReLU | ReLU |
| FFN dropout | 0 | 0 |
| Graph-embedding batch normalization | false | false |
| Batch size | 32 | 32 |
| Maximum epochs | 100 | 100 |
| Early stopping | minimum `val_loss`, patience 10 | same |
| Checkpoint selection | minimum `val_loss`, one checkpoint | same |

Chemprop's `BinaryClassificationFFN` trains with binary cross entropy with logits and returns sigmoid
probabilities for evaluation/inference. Its default reported metric is
`torchmetrics.classification.BinaryAUROC` through Chemprop's `BinaryAUROC` wrapper. Upstream does
not implement the Phase-1 secondary metrics, calibration, or threshold selection.

The model uses Adam with the Chemprop 2.1.0 defaults: initial learning rate `1e-4`; a stepwise
Noam-like schedule with two warmup epochs rising linearly to `1e-3`; then exponential decay to
`1e-4` by the configured final epoch. The seed is passed to `pl.seed_everything(seed)`. The Trainer
uses `accelerator="auto"`, `devices="auto"`, and logs every five steps. Exact deterministic CUDA
behavior is not guaranteed by these calls alone and must be strengthened and recorded locally.

For the model we intend to reproduce first, use the **MoleculeNet BBBP architecture**, not the B3DB
classification hyperparameters. The project adaptation remains a distinct `BBB_Martins` experiment.

## Feature normalization

Upstream constructs molecule datapoints and then calls
`train_dset.normalize_inputs("V_f")`. Chemprop uses scikit-learn `StandardScaler`, fitted to the
concatenation of every heavy-atom row in the training molecules. Each of the six GGL columns is
therefore centered and scaled independently, with atoms—not molecules—as observations. The same
fitted scaler is applied to validation and test features.

`MoleculeDatapoint.__post_init__` changes GGL `NaN` values to zero before scaler fitting. Thus an
atom with no supported retained neighbors can contribute zero-filled entries to training
statistics. Normalization happens before on-the-fly molecular graph construction; the normalized
six columns are then concatenated into the atom feature matrix.

Our implementation must fit this scaler on training atoms only, serialize its means, scales,
variance, feature order, scikit-learn version, training molecule/atom counts, and training input
hashes, and apply it without refitting to validation. No test feature or statistic may be loaded.

## Data adaptation without re-splitting or test access

### Local inputs available at inspection time

The configured prepared root is
`outputs/local/classification_expansion/coordinated/bbb_martins`, with `train.csv` and `valid.csv`.
Those ignored prepared files were not present in this Windows clone, so no real train/validation
rows were opened during this inspection. The project writer and configuration establish this schema:

```text
molecule_id,smiles,canonical_smiles,target,split
```

The configuration fixes the split-manifest ID to
`6028495d3792e5ad7a6ac1ffbadcaa6f1f0462ca0ca9f55a6b0babadcb76342b`, the task to binary
classification, and the positive class to BBB-permeable. The old test CSV was not opened.

Before implementation, run a metadata-only preflight against **train and validation only** to
confirm columns, row counts, hashes, binary labels, unique stable IDs, exact/scaffold separation,
and agreement with the manifest. Stop on any mismatch.

### Deterministic mapping

1. Load `train.csv` as training and `valid.csv` as validation. Do not concatenate and re-split them;
   do not invoke `data.make_split_indices`; do not resolve or load a test path.
2. Retain `molecule_id`, original `smiles`, `canonical_smiles`, `target`, and `split` in every
   manifest and prediction artifact. Assert the split column agrees with the source file.
3. Map `target` directly to one binary task; assert values are exactly `{0,1}` and record that `1`
   means BBB-permeable. No sign or class inversion is allowed.
4. Use the stored `canonical_smiles` as the identity/split/join key and proposed graph/geometry
   input; retain original `smiles` for provenance. Confirm the canonical string round-trips under
   the pinned RDKit without losing encoded stereochemistry. Any disagreement is a blocking report,
   not an automatic rewrite.
5. Map each molecule to an ASCII-safe geometry stem derived deterministically from its stable ID
   plus a canonical-SMILES hash. Store the mapping explicitly; do not assume arbitrary IDs are safe
   filenames.
6. Generate and validate geometry/features separately per split. Fit the GGL scaler on training
   atoms only, then transform validation. Never derive geometry settings, failure policy, or
   preprocessing statistics from validation performance.
7. Construct training and validation datasets directly from their respective records. Early
   stopping and checkpoint selection use validation loss only. Phase-1 threshold selection and any
   calibration use saved validation predictions only.
8. Export validation predictions keyed by `molecule_id` and `canonical_smiles` so comparison with
   Chemprop verifies the exact same molecules rather than only equal counts.

The coordinated preparation already deduplicates within endpoint by canonical structure and keeps
scaffolds in one split; the GMC adapter must recheck rather than change that membership. RDKit
preserves SMILES stereochemistry when encoded, and the ordinary bond features include bond stereo.
The adapter must record isomeric/canonicalization settings and must not discard stereochemistry.

Upstream code does not remove disconnected fragments; the paper says isolated ions were removed
during cleaning. For our fixed membership, do not silently choose a largest fragment or strip salts.
Flag disconnected records during preflight and require a documented scientific decision that
preserves the validation comparison. Likewise, invalid SMILES, unsupported SYBYL types, conformer
failures, atom-map failures, or non-finite features must be reported by molecule and reason. No
molecule may disappear through `zip` truncation or be silently excluded.

## Geometry and GGL reproducibility plan

For a fixed MOL2 file and fixed numeric/library versions, the checked-in GGL extractor has no random
operation and should be deterministic. Geometry creation is the unresolved stochastic component.

The paper reports:

- MoleculeNet BBBP: 2,039 compounds with pre-existing 3D structures;
- B3DB classification/regression: one low-energy 3D conformer per molecule;
- primary generation with OpenEye OMEGA; and
- fallback generation with RDKit and OpenBabel.

Neither the paper main text nor repository pins OMEGA/RDKit/OpenBabel versions, OMEGA options,
fallback order/details, embedding seed, force field, minimizer, iteration limit, energy selection,
protonation, tautomer handling, salt handling, MOL2 writer, or atom-order mapping. OMEGA is also a
separately licensed commercial dependency. Consequently, geometry for new `BBB_Martins` molecules
cannot yet be claimed as an exact reproduction.

Before geometry implementation, resolve this decision in writing:

- seek the authors' preprocessing script/settings or permission to use their exact implementation;
- determine whether an authorized OMEGA installation/license exists on ECHO; and
- if not, designate a pinned RDKit/OpenBabel fallback as a documented compatibility deviation.

The eventual geometry specification must pin the conformer algorithm and version, random seed,
hydrogen/protonation policy, force field and minimization settings, number of candidates, energy
selection/tie-breaking, coordinate precision, OpenBabel/MOL2 options, SYBYL typing, and failure
policy. It must hash the input identity and MOL2 output.

Hydrogens may be needed during conformer generation/minimization and may appear in MOL2, but the
checked-in GGL extractor drops exact SYBYL type `H`, while the RDKit graph is created without
explicit hydrogens. The final heavy-atom order must be proven equivalent between MOL2 and RDKit.
Conversion tools can reorder atoms, so the implementation should carry atom-map identifiers through
conversion and reorder GGL rows to RDKit order after a one-to-one check.

For source-faithful GGL, pin cutoff 12.0 Å, the chosen kernel row, the 45-type radii table, pairwise
distance precision, unsupported-type behavior, hydrogen filter, `NaN`-to-zero behavior, and six
statistics in their exact order. Store the original kernel index and resolved type, `tau`, and
`kappa`, not just the feature filename.

Steps likely to vary between runs or versions are conformer embedding, multi-conformer energy
ranking, minimization convergence, protonation/tautomer choice, aromaticity/SYBYL typing, atom order
during format conversion, RDKit canonical SMILES, floating-point distance/kernel calculations, and
CUDA training. Golden fixtures and hashes are required at each boundary.

## Environment compatibility

Upstream provides minimum/unbounded dependencies rather than a lock file. No CUDA version, GPU,
PyTorch build, or exact package solution is recorded.

| Component | Upstream evidence | Current ECHO/Chemprop context | Phase-1 recommendation |
|---|---|---|---|
| Python | `>=3.11` | Project preference/environment uses 3.11 | Pin 3.11 in a separate environment |
| Chemprop | Bundled 2.1.0 | 2.3.1 | **Version mismatch**; use 2.1.0 first for source reproduction |
| PyTorch | `>=2.1` | 2.6.0+cu124 | Nominally satisfies minimum, but unverified with Chemprop 2.1/Lightning; test and pin |
| CUDA | Not specified | RTX 6000 Ada, CUDA-capable, PyTorch CUDA 12.4 build | CUDA 12.4 is plausible, not an upstream reproduction fact; verify with a no-training smoke |
| Lightning | `>=2.0` | 2.6.5 in the existing GPU environment | Unbounded upstream requirement; 2.6 has compatibility handling added in later Chemprop, so do not assume 2.1 is safe |
| RDKit | Unpinned | Present in project environments | Pin; affects graph, stereo, conformers, canonicalization, and atom order |
| NumPy | `<2.0.0` | Chemprop 2.3.1 no longer imposes `<2` | Pin `<2` for upstream behavior |
| SciPy | Unpinned | Project-dependent | Pin; required for `cdist` |
| pandas | Unpinned | Project-dependent | Pin; CSV/MOL2 feature plumbing |
| scikit-learn | Unpinned | Project-dependent | Pin; `StandardScaler` |
| BioPandas | README extra dependency, unpinned | Not part of standard Chemprop dependency set | Add and pin; required for MOL2 parsing |
| OpenBabel | Paper fallback only; absent from repository requirements | Unknown | Resolve and pin only if selected for generation/MOL2 conversion |
| OpenEye OMEGA | Paper primary generator only | License/availability unknown | Treat as a licensing/availability gate, not an assumed dependency |
| astartes | Chemprop 2.1 dependency | Present transitively in Chemprop environments | Not used for local split assignment |
| torchmetrics | Lightning/Chemprop transitive dependency | Version coupled to Lightning | Pin because upstream AUROC behavior depends on it |
| ConfigArgParse, rich, descriptastorus | Chemprop 2.1 dependencies | Environment-dependent | Include only as required by the pinned Chemprop installation |

Chemprop 2.3.1 has changed internals, graph/data APIs, checkpoint-loading behavior, NumPy constraints,
and Lightning 2.6 compatibility code relative to 2.1.0. Some constructors still look source-compatible,
but running the upstream scripts in `admet-chemprop` would not reproduce the authors' declared
Chemprop version and could change serialization or runtime behavior. Create a separate Conda
environment, tentatively `admet-gmc-mpnn`, after reviewing the proposed pins. Do not modify or reuse
`admet-chemprop` for the first reproduction attempt.

## Upstream reproduction target

The first upstream reference should be **MoleculeNet BBBP**, because it is binary BBB permeability,
has a dedicated `train_bbbp.py`, and is the closest scientific match to `BBB_Martins`. Keep this
upstream reproduction separate from our fixed-split adaptation.

The paper reports MoleculeNet BBBP GMC-MPNN AUROC `0.947 ± 0.011` over five 8:1:1 scaffold splits
with seeds 0–4. It reports B3DB classification AUROC `0.9212 ± 0.0261` under the same high-level
five-split protocol. The repository's `test.py` hard-codes these MoleculeNet BBBP kernel indices by
seed: 266, 152, 953, 404, and 451. Their resolved `(type, tau, kappa)` values are:

| Seed | Kernel | Type | `tau` | `kappa` |
|---:|---:|---|---:|---:|
| 0 | 266 | exponential | 3.5 | 13.0 |
| 1 | 152 | exponential | 2.0 | 16.0 |
| 2 | 953 | Lorentz | 2.0 | 16.5 |
| 3 | 404 | exponential | 5.5 | 2.0 |
| 4 | 451 | exponential | 6.0 | 5.5 |

The code/repository does not document how those “best” kernels were selected or prove that selection
used validation rather than test. `train_bbbp.py` evaluates every feature file on its internal test
split; `train.py` averages test columns; and `test.py` evaluates its chosen kernels on test. These
facts prevent treating the repository workflow itself as a leakage-safe selection protocol.

An upstream reproduction may use the published fixed kernel list to check implementation behavior,
but it must report that the selection provenance is unresolved. Our `BBB_Martins` adaptation must
choose any kernel/hyperparameter using validation only under a predeclared rule and must never use
the old BBB test. Matching the upstream metric is a software/scientific sanity check, not a model
selection gate for our data.

## Proposed minimal local architecture

Do not copy the complete upstream repository or bundled Chemprop tree. Subject to the licensing
gate, implement only the project-owned adapters and the minimum GMC behavior:

```text
src/admet_platform/gmc_mpnn/
    features.py       # pinned GGL calculation and feature-manifest validation
    data.py           # fixed train/validation adapter and atom-order alignment
    model.py          # exact source-faithful BBBP architecture constructor
    training.py       # validation-only fit/checkpoint/prediction workflow
    inference.py      # checkpoint/scaler/feature loading and keyed predictions
configs/gmc_mpnn/
    bbb_martins.yaml
scripts/prepare_gmc_mpnn_features.py
scripts/run_gmc_mpnn_experiment.py
tests/gmc_mpnn/
```

Geometry generation should be isolated behind a deterministic, versioned interface rather than
hidden inside training. Raw/generated MOL2, `.npz`, caches, checkpoints, and predictions belong in
ignored output/cache paths. Configuration should refer to split files and manifests without any
test path for Phase-1 development runs.

Tests should cover kernel values against hand-calculated fixtures; the current covalent-mask
behavior; hydrogen and unsupported-type handling; six-column order; NaN handling; scaler train-only
fit; CSV/NPZ/atom identity alignment; stable hashes; fixed split preservation; disconnected
fragments; stereo round-trip; geometry failure reports; exact 78/14 model input shapes; model
hyperparameters; validation-only callbacks; and absence of test loading.

## Staged implementation sequence

1. **Environment specification.** Propose and review a separate pinned `admet-gmc-mpnn` Conda
   environment; resolve Chemprop 2.1.0, PyTorch/Lightning/CUDA compatibility, BioPandas, OpenBabel,
   OMEGA availability, and the GMC code-license gate.
2. **BBB train/validation adapter.** Load only the existing train/validation files, validate the
   manifest and schema, preserve stable identity, and emit no split or test behavior.
3. **GGL feature-generation reproduction.** Add source-faithful, independently tested kernel/MOL2
   behavior only after the licensing and geometry decisions are recorded.
4. **Preprocessing integrity tests.** Prove molecule/atom alignment, deterministic hashes,
   training-only scaling, failure accounting, and no leakage.
5. **Exact GMC-MPNN model construction.** Build the inspected MoleculeNet BBBP architecture without
   Chemprop-default substitutions and assert every resolved hyperparameter.
6. **Seed-13 smoke run.** Use a small train/validation-only subset to validate device, finite loss,
   checkpoint, inference, and artifacts.
7. **Seed-13 full validation run.** Train on the complete fixed training split and predict the
   complete fixed validation split.
8. **Inspect results.** Review learning curves, geometry coverage, reproducibility, all contracted
   metrics, calibration/threshold handling, and molecule-exact Chemprop comparison.
9. **Only then consider seeds 37, 73, 101, and 137.**

## Next Codex task only

Create and review the **environment specification only**: a separate proposed
`environment-gmc-mpnn.yml` (and synchronized dependency metadata if this repository's conventions
require it) plus a no-install compatibility checklist. Pin Python 3.11, Chemprop 2.1.0 behavior,
NumPy `<2`, and all directly required packages; document the candidate PyTorch/Lightning/CUDA
combination for the RTX 6000 Ada; and leave OpenEye OMEGA explicitly unresolved unless an authorized
license is confirmed. Do not install the environment, touch BBB data, generate geometry/features,
or implement model code in that next task.
