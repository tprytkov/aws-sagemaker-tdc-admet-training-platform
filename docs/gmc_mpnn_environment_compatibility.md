# GMC-MPNN Phase-1 environment and compatibility contract

## Decision and scope

Use the validated **strategy C** baseline:

1. establish a source-faithful baseline in an isolated Chemprop 2.1.0-compatible environment; then
2. only if needed, port already validated behavior to Chemprop 2.3.1 as a separately documented
   compatibility experiment.

This prioritizes scientific reproducibility over reuse of the existing `admet-chemprop`
environment. The validated environment is named `admet-gmc-mpnn` and has no dependency on that
environment.

The environment was successfully validated on ECHO without training and without locked-test
access. This document does not authorize BBB data access, geometry generation, feature generation,
model training, or locked-test access. The authoritative scientific and implementation constraints
remain
[`gmc_mpnn_bbb_phase1.md`](gmc_mpnn_bbb_phase1.md) and
[`gmc_mpnn_bbb_implementation_plan.md`](gmc_mpnn_bbb_implementation_plan.md).

## Validated ECHO environment

Validation used Git commit `71f5750f024af770387e9d8b15c35ec95b647958` on an NVIDIA RTX
6000 Ada Generation GPU.

| Component | Validated version |
|---|---|
| Python | `3.11.15` |
| Chemprop | `2.1.0` |
| Lightning | `2.1.4` |
| PyTorch | `2.1.2+cu121` |
| CUDA runtime | `12.1` |
| NumPy | `1.26.4` |
| scikit-learn | `1.9.0` |
| RDKit | `2024.09.6` |
| Setuptools | `80.10.2` |
| pytest | `8.4.2` |

The focused model/data suite completed with `18 passed`, and the full GMC-MPNN suite completed
with `222 passed`; neither run skipped a test.

The preserved environment provenance files have these SHA-256 hashes:

| Provenance file | SHA-256 |
|---|---|
| `environment_provenance.json` | `bb8d47347f337e86850617036fb2fd0d6b81e51a50cfc5ffd1dd6853d543e264` |
| `pip_freeze.txt` | `3fdb7b33798c2e26f3a0e5c2e7ab39200d5b865ffeafda032d7e435a667ea223` |
| `gpu_info.txt` | `197c3bd862ae505f5ad63da57c8e8075773928f82d53dcc40718371ed176559c` |

Setuptools is intentionally pinned to `80.10.2`. Lightning 2.1.4 imports `pkg_resources`; the
validated environment failed to import Lightning when tested with a newer Setuptools release that
removed `pkg_resources`. This pin is therefore part of the compatibility contract rather than an
incidental development dependency.

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

| Package/component | Upstream requirement | Validated ECHO version | Earlier Chemprop environment | Status | Evidence/notes |
|---|---|---|---|---|---|
| Python | `>=3.11` (constrained) | `3.11.15` | Python 3.11 | Passed | Recorded in environment provenance |
| PyTorch | `>=2.1` (constrained) | `2.1.2+cu121` | `2.6.0+cu124` | Passed | CUDA-enabled model/data tests passed |
| CUDA runtime/build | Not specified upstream | `12.1` through the `cu121` wheel | CUDA 12.4 wheel | Passed | Recorded in GPU provenance |
| NVIDIA GPU | Not specified upstream | RTX 6000 Ada Generation | RTX 6000 Ada Generation | Passed | Recorded in GPU provenance |
| torchvision | Not imported by the inspected BBBP/GGL path | Excluded | Not relevant | Low; adding it would introduce unnecessary Torch coupling | Confirm no import/runtime requirement appears in the smoke checks |
| Chemprop | Bundled `2.1.0` (pinned) | `2.1.0` | `2.3.1` | Passed | Exact model/data compatibility tests passed |
| Lightning | `>=2.0` (constrained) | `2.1.4` | `2.6.5` | Passed | Requires the Setuptools compatibility pin |
| RDKit | Imported/unspecified | `2024.09.6` | Environment-dependent | Passed | Geometry and atom-alignment tests passed |
| NumPy | `<2.0.0` (constrained) | `1.26.4` | Environment-dependent | Passed | Frozen feature and scaler tests passed |
| SciPy | `scipy.spatial.distance.cdist` imported; version unspecified | `<2`; exact lock pending | Project-dependent | Medium; NumPy ABI and distance calculations matter to GGL behavior | Import `cdist` and compare a tiny deterministic distance matrix |
| pandas | Imported/unspecified | `<3`; exact lock pending | Project-dependent | Medium; CSV dtypes and row ordering affect data/feature alignment | Record version and round-trip a synthetic table only |
| scikit-learn | Used by Chemprop normalization; unspecified | `1.9.0` | Project-dependent | Passed | Frozen-scaler tests passed |
| Setuptools | Build/runtime support; unspecified | `80.10.2` | Environment-dependent | Required | Retains `pkg_resources` for Lightning 2.1.4 |
| pytest | Development/testing dependency | `8.4.2` | Environment-dependent | Passed | 18 focused and 222 full GMC-MPNN tests passed |
| BioPandas | README dependency and `PandasMol2` direct import; unspecified | `<0.6`; exact lock pending | Not part of the standard Chemprop dependency set | High; MOL2 atom-table columns, ordering, and parser behavior are central to GGL input | Import `PandasMol2`; parse a small synthetic/approved MOL2 fixture and verify atom rows/types/coordinates |
| joblib | Not directly imported; transitive through scikit-learn | Resolver-selected, then locked | Transitive/project-dependent | Low to medium; affects serialization compatibility | Record resolved version; do not add direct use without need |
| torchmetrics | Transitive Chemprop/Lightning requirement; unspecified | Resolver-selected, then locked | Coupled to Lightning 2.6.5 | Medium; metric APIs and AUROC behavior may differ | Record version and confirm Chemprop metric construction only |
| astartes | Chemprop 2.1 dependency; not used for the fixed local split | Resolver-selected by Chemprop 2.1.0, then locked | Transitive | Low for this workflow; accidental use could re-split data | Confirm import resolution and that the adapter never invokes splitting |
| ConfigArgParse, rich, descriptastorus | Chemprop 2.1 dependencies; versions unspecified | Resolver-selected by Chemprop 2.1.0, then locked | Environment-dependent | Low to medium; install resolution may fail or vary | Record resolved versions and basic Chemprop import success |
| OpenEye OMEGA | Mentioned in the paper; absent from the inspected training path and repository requirements | Excluded | Availability/license unknown | Geometry policy and licensing are unresolved | Handle in the separate geometry-reproducibility task only |
| OpenBabel | Paper fallback only; absent from inspected training-path requirements | Excluded | Unknown | Geometry/MOL2 conversion may change typing and atom order | Add only if a later approved geometry policy requires it |
| Transformers / PyTDC | Not required by the inspected BBBP/GGL path | Excluded | Used elsewhere in the broader project | None for GMC-MPNN Phase 1 | Confirm no GMC module introduces these dependencies |
| Global RDKit descriptor tooling | Not used by upstream GMC-MPNN baseline | Excluded | May exist elsewhere in the project | Adding it would violate the descriptor-free Phase-1 contract | Reserve for the separately defined Phase 2 only |

## Validated environment rationale

[`environment-gmc-mpnn.yml`](../environment-gmc-mpnn.yml) records the exact core versions that
passed the no-training compatibility and GMC-MPNN test suites on ECHO:

- Python `3.11.15` matches the project's supported Python 3.11 patch while satisfying upstream
  `>=3.11`.
- Chemprop is pinned to `2.1.0`, the exact bundled upstream version. The upstream Chemprop subtree
  is not copied.
- PyTorch `2.1.2+cu121` passed on the ECHO RTX 6000 Ada with CUDA runtime 12.1. It remains a
  project compatibility selection rather than a version established by upstream.
- Lightning resolved to and passed at `2.1.4` when paired with Setuptools `80.10.2`.
- NumPy `1.26.4`, scikit-learn `1.9.0`, and RDKit `2024.09.6` passed the frozen preprocessing,
  scaler, atom-alignment, and model/data tests.
- pytest `8.4.2` is pinned as the development/testing version used for validation.

Because those packages are not fully pinned upstream, the YAML alone cannot recreate the authors'
unknown historical environment exactly. Reproduction instead uses the validated project environment
and the preserved `environment_provenance.json`, `pip_freeze.txt`, and `gpu_info.txt` hashes above.
The complete `pip_freeze.txt` remains authoritative for transitive packages not pinned directly in
the YAML. Any future dependency change requires a new compatibility validation and provenance set.

## CUDA strategy

PyTorch's official previous-version instructions publish a `2.1.2` CUDA 12.1 build. NVIDIA's CUDA
minor-version compatibility guidance states that CUDA 12.x requires a sufficiently recent driver.
The `2.1.2+cu121` build was validated directly on ECHO rather than inferred from another PyTorch
environment. References:

- [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

The ECHO RTX 6000 Ada successfully ran the compatibility validation with PyTorch `2.1.2+cu121`
and CUDA runtime 12.1. This is the validated project configuration, not a claim about the unknown
upstream training environment. Its GPU details are bound to the `gpu_info.txt` hash recorded above.

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

## Completed no-training compatibility validation

The ECHO validation recorded the exact core versions, the complete pip resolution, and GPU details
in the three hashed provenance files above. Chemprop 2.1.0 and its GMC-MPNN model/data interface
were exercised by both the focused and full GMC-MPNN suites. All 18 focused tests and all 222 full
suite tests passed with zero skips.

The validation did not train a model and did not authorize locked-test access. Future environment
changes must repeat these checks and produce new provenance hashes; they must not overwrite or be
represented as the validated environment recorded here.

## Decision gate

The compatibility gate has passed for the pinned Chemprop 2.1.0 model/data implementation and its
tests. It does not by itself authorize model training, Chemprop migration, dependency upgrades, or
locked-test access. Those actions require their own explicit workflow and provenance.
