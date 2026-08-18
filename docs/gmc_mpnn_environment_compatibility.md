# GMC-MPNN Phase-1 environment and compatibility contract

## Decision and scope

Use **strategy C**:

1. establish a source-faithful baseline in an isolated Chemprop 2.1.0-compatible environment; then
2. only if needed, port already validated behavior to Chemprop 2.3.1 as a separately documented
   compatibility experiment.

This prioritizes scientific reproducibility over reuse of the existing `admet-chemprop`
environment. The proposed environment is named `admet-gmc-mpnn` and has no dependency on that
environment.

This task specifies an environment and later verification only. It does not authorize environment
creation, package installation, BBB data access, geometry generation, feature generation, model
implementation, or training. The authoritative scientific and implementation constraints remain
[`gmc_mpnn_bbb_phase1.md`](gmc_mpnn_bbb_phase1.md) and
[`gmc_mpnn_bbb_implementation_plan.md`](gmc_mpnn_bbb_implementation_plan.md).

## Provenance and licensing constraint

The inspected upstream source is
[`MathIntelligence/GMC-MPNN-BBBP` at `ae080431950832be43e51f7cd9b0f7d4203e2267`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/tree/ae080431950832be43e51f7cd9b0f7d4203e2267).
Its bundled Chemprop subtree identifies as Chemprop 2.1.0 and carries the MIT license applicable to
that subtree.

> **Licensing gate:** The GMC-MPNN-specific source has no explicit license at the inspected
> revision. Environment recreation and scientific inspection are allowed. An independently
> written implementation of documented algorithms may be considered later. Upstream GMC-MPNN code
> must not be copied or vendorized into this repository until licensing is clarified. This records
> the absence of an explicit license; it is not a broader legal conclusion.

Requirements below are classified as:

- **pinned:** an exact version is stated or bundled upstream;
- **constrained:** upstream states a range but not an exact version;
- **imported/unspecified:** required by executed imports but not versioned upstream; or
- **inferred:** a local compatibility choice that must be tested and must not be represented as an
  upstream requirement.

## Compatibility matrix

| Package/component | Upstream requirement | Proposed version | Current Chemprop environment | Risk | Verification needed |
|---|---|---|---|---|---|
| Python | `>=3.11` (constrained) | `3.11.15` (project-supported exact patch) | Python 3.11 | Low; patch is local rather than upstream-pinned | Record `python --version`; run all import checks |
| PyTorch | `>=2.1` (constrained) | `2.1.2+cu121` (inferred candidate) | `2.6.0+cu124` | Medium; older Torch/extension behavior and a different CUDA runtime | Confirm version, CUDA runtime, tensor operation, and device access without training |
| CUDA runtime/build | Not specified upstream | PyTorch CUDA 12.1 wheel (`cu121`), inferred | CUDA 12.4 wheel on Linux ECHO | Medium; repository metadata cannot establish the server driver/runtime pairing | Confirm ECHO driver, `torch.version.cuda`, `torch.cuda.is_available()`, and a small GPU tensor operation |
| NVIDIA GPU | Not specified upstream | RTX 6000 Ada Generation on ECHO | RTX 6000 Ada Generation | Low to medium; hardware is capable, but environment access is unproved | Record `torch.cuda.get_device_name(0)` and device capability |
| torchvision | Not imported by the inspected BBBP/GGL path | Excluded | Not relevant | Low; adding it would introduce unnecessary Torch coupling | Confirm no import/runtime requirement appears in the smoke checks |
| Chemprop | Bundled `2.1.0` (pinned) | `2.1.0` | `2.3.1` | High; APIs, defaults, checkpoint behavior, NumPy constraints, and Lightning compatibility changed | Confirm exact version and construct the expected objects without fitting |
| Lightning | `>=2.0` (constrained) | `>=2.0,<2.2` (inferred first-test band) | `2.6.5` | High; upstream has no lock and later Chemprop contains newer Lightning compatibility work | Record resolved patch; import Trainer/callbacks; construct Trainer with training disabled |
| RDKit | Imported/unspecified | `<2025` provisional guard; exact lock pending | Installed in project environments; exact current value must be recorded on ECHO | High; parsing, aromaticity, stereochemistry, canonicalization, and atom order can vary | Record version; test known SMILES parsing and later approved atom-order fixtures |
| NumPy | `<2.0.0` (constrained) | `<2.0.0`; exact lock pending | Chemprop 2.3.1 environment does not impose the upstream cap | Medium; array semantics and binary compatibility affect SciPy and feature archives | Record resolved version; import with SciPy; exercise finite array calculations |
| SciPy | `scipy.spatial.distance.cdist` imported; version unspecified | `<2`; exact lock pending | Project-dependent | Medium; NumPy ABI and distance calculations matter to GGL behavior | Import `cdist` and compare a tiny deterministic distance matrix |
| pandas | Imported/unspecified | `<3`; exact lock pending | Project-dependent | Medium; CSV dtypes and row ordering affect data/feature alignment | Record version and round-trip a synthetic table only |
| scikit-learn | Used by Chemprop normalization; unspecified | `<2`; exact lock pending | Project-dependent | Medium; `StandardScaler` behavior and serialization are version-sensitive | Import and run a tiny deterministic `StandardScaler` example |
| BioPandas | README dependency and `PandasMol2` direct import; unspecified | `<0.6`; exact lock pending | Not part of the standard Chemprop dependency set | High; MOL2 atom-table columns, ordering, and parser behavior are central to GGL input | Import `PandasMol2`; parse a small synthetic/approved MOL2 fixture and verify atom rows/types/coordinates |
| joblib | Not directly imported; transitive through scikit-learn | Resolver-selected, then locked | Transitive/project-dependent | Low to medium; affects serialization compatibility | Record resolved version; do not add direct use without need |
| torchmetrics | Transitive Chemprop/Lightning requirement; unspecified | Resolver-selected, then locked | Coupled to Lightning 2.6.5 | Medium; metric APIs and AUROC behavior may differ | Record version and confirm Chemprop metric construction only |
| astartes | Chemprop 2.1 dependency; not used for the fixed local split | Resolver-selected by Chemprop 2.1.0, then locked | Transitive | Low for this workflow; accidental use could re-split data | Confirm import resolution and that the adapter never invokes splitting |
| ConfigArgParse, rich, descriptastorus | Chemprop 2.1 dependencies; versions unspecified | Resolver-selected by Chemprop 2.1.0, then locked | Environment-dependent | Low to medium; install resolution may fail or vary | Record resolved versions and basic Chemprop import success |
| OpenEye OMEGA | Mentioned in the paper; absent from the inspected training path and repository requirements | Excluded | Availability/license unknown | Geometry policy and licensing are unresolved | Handle in the separate geometry-reproducibility task only |
| OpenBabel | Paper fallback only; absent from inspected training-path requirements | Excluded | Unknown | Geometry/MOL2 conversion may change typing and atom order | Add only if a later approved geometry policy requires it |
| Transformers / PyTDC | Not required by the inspected BBBP/GGL path | Excluded | Used elsewhere in the broader project | None for GMC-MPNN Phase 1 | Confirm no GMC module introduces these dependencies |
| Global RDKit descriptor tooling | Not used by upstream GMC-MPNN baseline | Excluded | May exist elsewhere in the project | Adding it would violate the descriptor-free Phase-1 contract | Reserve for the separately defined Phase 2 only |

## Proposed environment rationale

[`environment-gmc-mpnn.yml`](../environment-gmc-mpnn.yml) is a conservative **candidate
specification**, not yet a final scientific lock:

- Python `3.11.15` matches the project's supported Python 3.11 patch while satisfying upstream
  `>=3.11`.
- Chemprop is pinned to `2.1.0`, the exact bundled upstream version. The upstream Chemprop subtree
  is not copied.
- PyTorch `2.1.2+cu121` is an inferred compatibility candidate: it stays in the upstream minimum
  2.1 release line and uses an officially published CUDA 12.1 Linux wheel. It is not an upstream
  observed version.
- Lightning is limited to the 2.1 release line for the first compatibility attempt because upstream
  states only `>=2.0`. Its resolved patch must be tested; the range is not evidence of the authors'
  environment.
- NumPy preserves the explicit upstream `<2.0.0` constraint. Other scientific packages receive
  broad major-version guards only to avoid unconstrained future major releases. Upstream does not
  justify exact pins for them.

Because those packages are not fully pinned upstream, the YAML alone cannot recreate the authors'
unknown environment exactly. Before any scientific work, a later no-training environment task must
solve it once, run the checks below, review the resolved versions, and export both a Conda explicit
specification and `python -m pip freeze`. The reviewed resolutions should then become the immutable
environment lock used for runs. If resolution or checks fail, change one compatibility variable at
a time and document the reason; do not silently fall back to current/latest packages.

## CUDA strategy and uncertainty

PyTorch's official previous-version instructions publish a `2.1.2` CUDA 12.1 build. NVIDIA's CUDA
minor-version compatibility guidance states that CUDA 12.x requires a sufficiently recent driver;
the ECHO driver version must be checked directly rather than inferred from the currently working
PyTorch 2.6.0/cu124 environment. References:

- [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

The RTX 6000 Ada is a reasonable target for the proposed CUDA-enabled build, but repository
metadata contains no CUDA toolkit, driver, GPU, or PyTorch build record. Therefore
`torch==2.1.2+cu121` is a testable local proposal, not a claim about upstream. Do not reuse
`torch==2.6.0+cu124` merely because it works for Chemprop 2.3.1. If cu121 fails the no-training GPU
probe, stop and document the observed driver/runtime error before proposing another build.

## RDKit, MOL2, and geometry are separate concerns

The workflow has three distinct layers:

1. **RDKit/Chemprop molecular parsing and ordinary graph features:** RDKit parses SMILES and affects
   atom order, aromaticity, stereochemistry, canonicalization, and the ordinary molecular graph.
2. **MOL2/BioPandas input:** BioPandas reads pre-existing MOL2 atom rows containing coordinates and
   SYBYL atom types used by GGL calculations. Its row ordering must later be proven to match the
   RDKit heavy-atom order.
3. **Geometry generation:** creation, optimization, selection, and MOL2 serialization of conformers
   is not reproducibly specified or implemented in the inspected repository.

OMEGA and OpenBabel are therefore not dependencies of this environment. The paper's mention of
them does not establish a versioned training-path requirement. Geometry generation is deferred to
a separate task after its reproducibility, licensing, atom-mapping, and failure policies are
decided.

## No-training compatibility checklist

After explicit approval to create the environment, perform these checks on the Linux ECHO server.
They must not read BBB files, generate geometry/features, or fit a model.

- [ ] Create only from `environment-gmc-mpnn.yml`; record the solver, channels, and complete solve
  output.
- [ ] Record `python --version`, `conda list`, `conda list --explicit`, and
  `python -m pip freeze`.
- [ ] Import `torch`, `chemprop`, `lightning`, `rdkit`, `numpy`, `scipy`, `pandas`, `sklearn`, and
  `biopandas` in one clean Python process.
- [ ] Confirm PyTorch is exactly the proposed 2.1.2 CUDA build; record `torch.version.cuda`.
- [ ] Confirm `torch.cuda.is_available()` and record the GPU name, device capability, driver
  information, and a successful small tensor operation on `cuda:0`.
- [ ] Confirm Chemprop reports `2.1.0` and Lightning reports a resolved `2.0.x` or `2.1.x` version.
- [ ] Record RDKit, NumPy, SciPy, pandas, scikit-learn, BioPandas, joblib, torchmetrics, and all
  Chemprop transitive dependency versions.
- [ ] Construct, but do not fit, the Chemprop 2.1 featurizer, message-passing, aggregation,
  binary-classification FFN, and MPNN objects required by the authoritative plan. Verify expected
  input dimensions and parameter shapes; do not save a checkpoint.
- [ ] Parse a tiny synthetic or explicitly approved non-BBB MOL2 fixture with `PandasMol2`; verify
  atom-row order, `x/y/z`, and SYBYL `atom_type` columns. Do not generate the MOL2 file with a
  geometry tool.
- [ ] Run a minimal RDKit parse on a synthetic public SMILES and record atom order and canonical
  isomeric SMILES behavior.
- [ ] Confirm no training loop, optimizer step, BBB data read, geometry generation, GGL generation,
  old BBB test access, checkpoint, or run artifact occurs.

Passing imports alone is insufficient: the resolved lock is viable only if construction, MOL2
parsing, and CUDA probes pass together and their output is captured in a public-safe compatibility
record.

## Decision gate

Proceed only after the candidate environment resolves, passes every no-training check, and its
reviewed exact dependency lock is recorded. A failure requires a documented compatibility revision;
it does not authorize a Chemprop 2.3.1 port, geometry implementation, or model training.

If the environment specification is viable, the **next implementation task** is:

> **Implement a read-only BBB train/validation data adapter and split-integrity tests, without GGL
> feature generation or model code.**

That task must not resolve, open, inspect, summarize, or otherwise access the old BBB test file.
