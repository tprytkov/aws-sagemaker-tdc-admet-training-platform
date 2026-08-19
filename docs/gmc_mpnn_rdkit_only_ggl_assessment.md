# RDKit-only assessment for the released GMC-MPNN six-feature GGL representation

## Scope, evidence, and decision

This is an inspection and design record. No BBB data were resolved or accessed, no geometry was
generated, no package was installed, and no model or training code was implemented.

The source target was the public `MathIntelligence/GMC-MPNN-BBBP` repository at commit
[`ae080431950832be43e51f7cd9b0f7d4203e2267`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/tree/ae080431950832be43e51f7cd9b0f7d4203e2267).
The inspected executable path was:

```text
CSV id
-> <id>.mol2
-> utils/get_ggl_ligand_features.py
-> PandasMol2 atom table
-> utils/ggl_ligand.py
-> one (n_heavy_atoms, 6) array
-> positional NPZ entry arr_i
-> train_bbbp.py / train_b3db_*.py
-> Chemprop MoleculeDatapoint(..., V_f=array)
```

Primary evidence:

- pinned [`utils/ggl_ligand.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/utils/ggl_ligand.py),
  [`utils/get_ggl_ligand_features.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/utils/get_ggl_ligand_features.py),
  and [`utils/ligand_SYBYL_atom_types.csv`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/utils/ligand_SYBYL_atom_types.csv);
- pinned [`train_bbbp.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/train_bbbp.py),
  [`train_b3db_cls.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/train_b3db_cls.py),
  and [`train_b3db_regression.py`](https://github.com/MathIntelligence/GMC-MPNN-BBBP/blob/ae080431950832be43e51f7cd9b0f7d4203e2267/train_b3db_regression.py);
- the repository's only public branch and its full reachable history; and
- [arXiv:2507.18926v5](https://arxiv.org/abs/2507.18926) and the
  [journal supplementary information](https://www.rsc.org/suppdata/d5/me/d5me00175g/d5me00175g1.pdf).

### Decision

**Option A: RDKit-only GMC-MPNN preprocessing.**

The exact scientific reason is that the released six-column extractor needs Cartesian coordinates
and the source radius associated with each **element**, but it does not use any information that
distinguishes SYBYL subtypes. `C.1`, `C.2`, `C.3`, `C.ar`, and `C.cat` all map to 1.70 Angstrom;
all seven nitrogen types map to 1.55; all three oxygen types map to 1.52; and all four sulfur types
map to 1.80. Every other supported element has only one listed type (apart from the same principle
for `P.3`). Atomic number therefore determines the same radius exactly over the supported domain.
Fine-grained SYBYL perception adds no numerical information to the public six-feature path.

The source does use the MOL2 `atom_type` string as a lookup key, so it would be inaccurate to say
that the column is never referenced. The scientifically important result is narrower and decisive:
**among the released accepted labels, the subtype portion of the SYBYL label never changes a
radius, a kernel, a mask, a selected subgraph, or any output value.** OpenBabel is consequently not
required for this representation.

### Effect on the earlier Phase-1 plans

This focused source audit corrects the earlier inference in
`gmc_mpnn_geometry_ggl_plan.md` and `gmc_mpnn_bbb_implementation_plan.md` that aromatic, amide,
charged, and other SYBYL subtypes change GGL through different radii. The source table proves that
they do not. The geometry, leakage, scaling, Chemprop-interface, licensing, and fail-closed
requirements in those documents remain applicable. Their proposed OpenBabel/SYBYL boundary and
associated atom-remapping work should be replaced by the direct RDKit-order design in this
assessment before implementation begins.

## 1. Exact public BBBP and B3DB path

`get_ggl_ligand_features.py` reads CSV columns `id` and `smiles`, but only `id` affects feature
generation. For each ID it opens `<data_folder>/<id>.mol2`, selects a kernel row, calls
`GGL_LIGAND.get_atom_features`, and saves the returned arrays in CSV order with `numpy.savez`.
The read `smiles` list is unused.

`ggl_ligand.py` asks BioPandas for the MOL2 atom table. It filters rows whose `atom_type` equals
exactly `H`, then constructs a smaller dataframe containing:

- `ATOM_INDEX` from `atom_id`;
- misleadingly named `ATOM_ELEMENT` from the full SYBYL `atom_type`; and
- `X`, `Y`, and `Z` from the coordinate columns.

After construction, `ATOM_INDEX` is never referenced. `ATOM_ELEMENT` is used to form all ordered
type-pair strings and look up the sum of their table radii. Coordinates produce an all-pairs
Euclidean distance matrix. No MOL2 bond table or other atom-table field is used.

The six columns are written in this exact order:

```text
minimum, maximum, sum, mean, median, population_standard_deviation
```

`train_bbbp.py` loads `arr_0`, `arr_1`, and so on, zips them positionally with RDKit molecules and
targets, and supplies each array as Chemprop `V_f`. It fits the `V_f` scaler on training atoms and
applies it to validation and test. Both public B3DB training scripts use the same loading,
`MoleculeDatapoint`, featurizer, and train-only scaling path. They do not contain an alternative
colored-subgraph extractor.

## 2. Explicit MOL2 dependency table

“Required” below refers to the public **six-feature mathematics**, not what the current parser
happens to demand syntactically.

| Input field | Read from MOL2? | Used numerically? | Used only for filtering/alignment? | Required for six-feature GGL? |
|---|---:|---:|---:|---:|
| x/y/z coordinates | Yes | Yes: Euclidean `d_ij` | No | **Yes** |
| SYBYL atom type | Yes | As a radius-table lookup key in the released code; the subtype carries no numerical information | Exact `H` filtering and supported-type gating | **No as SYBYL typing**; elemental identity is sufficient |
| Element / atomic identity | No separate MOL2 field; encoded by `atom_type` | Yes: element determines the source radius | Identifies hydrogen/heavy atoms | **Yes** |
| Atom index (`atom_id`) | Yes, copied to `ATOM_INDEX` | No | No; even alignment does not consult it | No |
| Atom-table row order | Implicitly yes | Only determines output row order; permuting all atom records consistently merely permutes output rows | **Yes**, positional alignment with Chemprop | Required only for row correspondence, not values for a mapped atom |
| Atom name | Present in MOL2 but not selected | No | No | No |
| Formal charge | Not read as a distinct field | No | No | No |
| Partial charge | Present in typical MOL2 atom rows but not selected | No | No | No |
| Bond type | Bond table is not read | No | No | No |
| Aromaticity | Not read separately | No; `*.ar` has the same radius as other subtypes of that element | No | No |
| Residue/substructure identifier or name | Present in typical MOL2 atom rows but not selected | No | No | No |

Bond order, aromaticity, formal/partial charge, atom name, and substructure fields could influence
an external program's assignment of a SYBYL label, but that indirect typing work is immaterial here
because all listed subtypes of the resulting element have the same radius. They remain important to
Chemprop's ordinary 2D atom/bond features, which are a separate input branch constructed by RDKit.

## 3. Exact mathematics of the six features

Let the retained rows be atoms `i = 1,...,n`, with coordinate `x_i`, element-derived source radius
`r_i`, and

```text
d_ij = ||x_i - x_j||_2
R_ij = r_i + r_j
```

For configured scale `tau` and power `kappa`, the public kernels are:

```text
exponential: w_ij = exp(-(d_ij / (tau * R_ij)) ** kappa)
Lorentz:     w_ij = 1 / (1 + (d_ij / (tau * R_ij)) ** kappa)
```

Thus a weight is not a function of distance alone: it is a function of distance, the two
**elemental** radii, and the globally selected kernel parameters. It is not modified by SYBYL
hybridization/aromatic/amide/charge subtype.

The implementation computes the following row-dependent quantity:

```text
S_i = sum_k R_ik = sum_k (r_i + r_k)
```

It then retains finite off-diagonal weights satisfying:

```text
i != j
d_ij <= cutoff
d_ij < S_i
```

The last condition is the literal behavior of the variable named `covalent_bond_mask`:
`distances >= pairwise_radii.sum(axis=1).reshape(-1, 1)` is masked. It is not the pairwise
covalent/non-covalent rule described by the paper and is usually nonbinding for ordinary connected
molecules because `S_i` grows with atom count. A source-faithful reproduction must preserve this
surprising row-sum mask and test it explicitly; it must not silently replace it with `d_ij >= R_ij`
or with molecular bond information.

For each central atom `i`, let `K_i` be the multiset of its retained finite `w_ij`. The output is:

| Feature | Exact summarized set and operation |
|---|---|
| minimum | `min(K_i)` via `numpy.nanmin` |
| maximum | `max(K_i)` via `numpy.nanmax` |
| sum | `sum(K_i)` via `numpy.nansum` |
| mean | arithmetic mean of `K_i` via `numpy.nanmean` |
| median | median of `K_i` via `numpy.nanmedian` |
| population standard deviation | `sqrt(mean((w - mean(K_i))^2))` via `numpy.nanstd(ddof=0)` |

An empty row yields zero for `nansum` and `NaN` for the other five summaries. Chemprop later
zero-fills `V_f` NaNs. A robust adapter should report this condition instead of allowing it to be
silent.

Direct answers about the calculation:

- Kernel weights depend on pairwise distance, element-specific radius sums, and global
  `tau`/`kappa`; they are not functions of coordinates alone.
- Fine-grained atom types do not modify weights because every subtype of an element has one common
  radius.
- The public extractor does not select atom-type-specific subgraphs. Every supported retained atom
  is considered in one pooled matrix.
- It does not choose different kernels for different atom or SYBYL types. One global kernel family
  and parameter pair applies to the entire molecule.
- The 12 Angstrom cutoff is purely geometric and type-independent.
- No charge, bond, aromaticity, residue, atom-name, or cheminformatics descriptor enters these six
  columns.

### Radius information that actually survives SYBYL parsing

Grouping the released 45 type rows by chemical element produced no element with more than one
unique radius:

| Element | Released accepted labels | Radius (Angstrom) |
|---|---|---:|
| C | `C.1`, `C.2`, `C.3`, `C.ar`, `C.cat` | 1.70 |
| N | `N.1`, `N.2`, `N.3`, `N.4`, `N.am`, `N.ar`, `N.pl3` | 1.55 |
| O | `O.2`, `O.3`, `O.co2` | 1.52 |
| S | `S.2`, `S.3`, `S.o`, `S.o2` | 1.80 |
| P | `P.3` | 1.80 |
| H | `H` | 1.20 (filtered before calculation) |
| Other supported elements | one label each | `As` 1.85; `B` 0.85; `Be` 1.53; `Br` 1.85; `Cl` 1.75; `Co` 2.00; `Cu` 1.28; `F` 1.47; `Fe` 1.26; `Hg` 1.50; `I` 1.98; `Ir` 2.00; `Mg` 1.73; `Os` 2.00; `Pt` 1.75; `Re` 2.05; `Rh` 2.00; `Ru` 2.05; `Sb` 2.06; `Se` 1.90; `Si` 2.10; `Te` 1.40; `V` 1.34; `Zn` 1.39 |

The RDKit-only implementation must use an explicit, versioned atomic-number-to-**released-radius**
mapping. It must not substitute whatever van der Waals radii a particular RDKit build returns.

## 4. Paper representation versus public source

The paper describes a richer WCS representation: atoms are separated into colored subgraphs by
atom-type pairs, and each atom receives statistics for its corresponding type-specific subgraphs.
It discusses 12 broad elements, while its ablation discussion also names fine-grained pairs such as
`N.ar-O.2` and `N.pl3-O.2`. That representation preserves which partner atom type contributed each
interaction.

The public extractor does not implement this separation. It collapses every retained atom into one
all-to-all weight matrix and emits only six total values per heavy atom. Searches of the public
branch, the complete reachable repository history, the current and deleted feature-extraction
paths, BBBP scripts, and B3DB scripts found no implementation of the paper's type-specific WCS
feature construction or atom-pair ablation. Earlier/deleted extractor copies contain the same
pooled calculation.

| Representation | Paper | Public BBBP source | Public B3DB source | Can reproduce from public revision? |
|---|---|---|---|---|
| Type-pair colored WCS subgraphs with statistics retained per partner type | Described | Not implemented or loaded | Not implemented or loaded | **No source-faithful reproduction**; relevant implementation/settings are missing or unreleased |
| Fine-grained atom-pair ablation (for example `N.ar-O.2`) | Reported | No ablation path | No ablation path | No |
| Pooled six-value matrix over all supported heavy atoms | Figure/text broadly mention GGL summaries, but this loses the paper's colored channels | **Implemented and loaded as `V_f`** | **Same implementation and `V_f` path** | **Yes**, for fixed coordinates, released radii, mask, cutoff, and kernel |
| Ordinary RDKit/Chemprop 2D atom and bond features | Described as CAF/bond inputs | Implemented by Chemprop | Implemented by Chemprop | Yes; separate from the six GGL columns |

Phase 1 can reproduce the public pooled six-feature implementation. It must not call that output a
reproduction of the paper-only colored-subgraph channels or use the paper's colored-subgraph claims
to characterize what those six columns contain.

## 5. Is MOL2 necessary?

### Is MOL2 mathematically required?

**No.** The released six values can be computed directly from an RDKit conformer's heavy-atom
coordinate matrix plus atomic numbers, using the released element radii and exact public masks and
kernels.

```text
RDKit conformer
-> heavy-atom coordinates and atomic numbers in RDKit order
-> released atomic-number-to-radius mapping
-> identical distance matrix
-> identical row-sum and 12 Angstrom masks
-> identical kernel matrix
-> identical six row statistics
```

### Is MOL2 operationally required by the upstream program?

**Yes, but only as a legacy parser boundary.** `ggl_ligand.py` accepts a MOL2 path and uses
BioPandas, so that exact script cannot run without MOL2. Neither its output nor the training script
contains a mathematical dependence on MOL2 serialization. A clean-room RDKit-coordinate function
can remove the file-format boundary while preserving the calculation.

Avoiding serialization also avoids coordinate rounding introduced by a MOL2 writer. Equivalence
must compare calculations using numerically identical coordinate arrays; a comparison that writes
fewer decimal places is partly a test of serialization precision, not of the GGL algorithm.

## 6. Is SYBYL typing necessary?

- **Is SYBYL referenced?** Yes. The released parser uses the full string as a table key and uses
  exact `H` for filtering.
- **Does fine-grained SYBYL information affect the final six values?** No. Subtypes within an
  element have identical table radii, no subtype-specific subsets are created, and no other subtype
  property is used.
- **Is it loaded but unused?** The string is not wholly unused; it is an unnecessarily fine-grained
  way to obtain element radius and recognize hydrogen. The part beyond elemental identity is
  numerically unused.
- **Can RDKit atomic number replace it?** Yes, exactly over the released supported domain, provided
  the implementation maps atomic number to the released radius table and rejects unsupported
  elements explicitly.
- **Can SYBYL typing change which atoms contribute?** Not among valid supported heavy-atom labels.
  All supported heavy atoms enter the one pooled matrix. The only normal filter is exact `H`.

There are two malformed/unsupported-input edge cases to document rather than emulate silently:

1. A hydrogen given a non-`H` type is retained by the public exact-string filter and then normally
   becomes unsupported/`NaN`. RDKit atomic-number filtering would correctly remove it. Valid
   source-compatible MOL2 uses `H`, and Chemprop's expected row count is the heavy-atom count.
2. An unsupported type receives `NaN` radii and is omitted by NaN-aware summaries, potentially
   degrading every row silently. An RDKit-only implementation should fail with an unsupported
   atomic number instead of treating this accidental failure behavior as a chemical feature.

These fail-closed differences do not change valid-domain numerical equivalence. They must be named
in the implementation contract and tested.

## 7. Hydrogen handling

The released extractor removes MOL2 rows only when `atom_type == "H"` before distances, radii,
kernels, masks, or statistics are computed. Therefore hydrogen coordinates and H-heavy or H-H
weights do not contribute to the public six features. Training builds the Chemprop molecule with
`keep_h=False, add_h=False`, so `V_f` is expected to have one row per RDKit heavy atom.

An equivalent RDKit-only pipeline should:

1. add explicit hydrogens for embedding and force-field optimization;
2. preserve original heavy-atom indices through `AddHs` and conformer selection;
3. extract only atoms whose atomic number is not 1 after geometry is finalized;
4. keep those atoms in original RDKit heavy-atom order; and
5. compute distances and GGL values only on that heavy-atom matrix.

Hydrogen isotopes also have atomic number 1 and should be excluded. A synthetic test must assert
that adding/removing the explicit hydrogen rows after fixing the heavy coordinates does not change
GGL output.

## 8. Atom ordering

For a fixed central atom, the statistics are invariant to the order of all partner atoms. A
consistent permutation of atom records permutes rows and columns of the distance/kernel matrices
and produces the same corresponding feature vectors in permuted row order. Thus MOL2 atom order
does not change the chemistry of the calculation; it matters only because output row `i` is
positionally concatenated to Chemprop atom `i`.

The public workflow assumes, but does not prove, this positional correspondence. It does not sort
or map using `atom_id`, element, name, bonds, or coordinates. Direct computation in original RDKit
heavy-atom order makes the correspondence true by construction and eliminates:

- RDKit-to-OpenBabel atom-order conversion risk;
- MOL2 row remapping and graph-isomorphism ambiguity;
- BioPandas row-order dependence; and
- a second chemistry parser's perception and serialization boundary.

The implementation must still assert that `AddHs`, embedding, optimization, and conformer
selection preserve stored original heavy indices and that the final `(n_heavy, 6)` row count and
atomic-number sequence match the Chemprop molecule.

## 9. RDKit-only Phase-1 pipeline

| Step | Classification relative to released behavior | Required record or qualification |
|---|---|---|
| Canonical isomeric SMILES -> RDKit molecule | RDKit is the same molecular-identity/ordinary-graph authority used by training; canonicalization is a local identity safeguard | Record input and canonical isomeric SMILES, RDKit version, and original heavy indices; do not silently standardize identity |
| Add explicit H | Geometry-generation step outside the public extractor; consistent with producing a physical conformer | Record hydrogen policy and prove heavy-index preservation |
| Deterministic ETKDGv3 geometry | **Documented deviation** from the authors' unreleased geometry workflow (OMEGA primary; paper fallback used RDKit ETKDGv2 before PDB/OpenBabel) | Record every parameter, seed, RDKit version, candidate count, and status |
| MMFF94s optimization / documented UFF fallback | **Documented deviation**; complete author force-field/selection settings and generated geometries are unavailable | Record parameter coverage, convergence, energy, fallback reason, and selection rule |
| Select one conformer deterministically | Public GGL expects one geometry, but the author selection procedure is not reproducible from the repository | Use a predeclared label-independent rule and record candidates/tie break |
| Extract heavy coordinates in original RDKit order | Equivalent replacement for MOL2 parsing and exact-`H` filtering | Assert original indices, atomic numbers, coordinate units, finite values, and row count |
| Map atomic numbers to released element radii | Numerically identical replacement for the SYBYL-key radius lookup on supported inputs | Version and hash the explicit mapping; do not use generic RDKit radii; fail on unsupported elements |
| Pairwise distance matrix | Identical mathematics | Use float64 and record units |
| Public kernel, row-sum mask, diagonal mask, and cutoff | Identical mathematics | Preserve the literal mask, selected global kernel family, `tau`, `kappa`, and 12 Angstrom cutoff |
| Six raw row summaries | Identical mathematics and column order | Preserve NumPy population-standard-deviation and empty-row behavior, while reporting invalid rows |
| Training-only feature scaling | Identical training behavior | Fit six columns on training atoms only and serialize scaler provenance |
| Chemprop 2.1.0 `V_f` | Identical public model interface | Assert `(n_RDKit_heavy_atoms, 6)` and exact row identity before concatenation |

This is a reproduction of the **released six-feature GMC representation**, not an exact
reproduction of author coordinates, published predictions, or the richer paper-only WCS
representation.

## 10. Future equivalence test design

The next implementation task should use five synthetic public molecules only. For each molecule,
start with one RDKit conformer and preserve one authoritative heavy-atom order and coordinate
matrix. Create two paths:

### A. Upstream-style MOL2 path

1. In a temporary verification environment, serialize the same coordinates to MOL2 with enough
   precision and valid supported SYBYL labels.
2. Record the explicit RDKit-heavy-index to MOL2-row map.
3. Run the pinned upstream extractor without copying it into this repository.
4. Reorder its result to authoritative RDKit heavy order if the writer changed order.

### B. Direct RDKit-coordinate path

1. Select the same heavy atoms in authoritative RDKit order.
2. Use the exact coordinates that path A's parser observed (or assert serialization preserved the
   original values within a predeclared bound).
3. Map atomic numbers to the released element radii.
4. Apply the independently written source-faithful calculation.

For each molecule and one or more boundary-focused synthetic coordinate fixtures, assert:

- identical heavy-atom count and ordered atomic-number sequence after explicit mapping;
- identical coordinate matrix for the algorithm comparison;
- pairwise distance matrices equal at `rtol=1e-12`, `atol=1e-12` when coordinate bytes match;
- identical `R_ij` radius-sum matrix;
- identical diagonal, cutoff, row-sum, finite, and combined retained-pair masks;
- identical exponential/Lorentz kernel values under the chosen test settings;
- identical `(n_heavy, 6)` matrix at `rtol=1e-12`, `atol=1e-12`; and
- intentional atom-row permutation is either correctly mapped or detected.

Include targeted invariance tests in which `C.2` is changed to `C.ar`, `N.am` to `N.pl3`, or
`O.2` to `O.co2` without changing element or coordinates: output must be bitwise/equivalently
unchanged. Include a negative test changing an element (for example C to N): radii and output must
change. Include a cutoff-boundary and row-sum-mask fixture so comparison does not pass merely
because both masks are inactive on ordinary molecules.

OpenBabel is not installed or required in the production environment. If desired for independent
verification, later create a disposable, pinned Conda environment containing OpenBabel and
BioPandas, generate only the five synthetic MOL2 fixtures, capture versions/hashes, run the
comparison, and discard the environment. Passing that audit validates removal of a legacy format
boundary; it does not make OpenBabel a Phase-1 dependency. A hand-authored minimal MOL2 fixture can
also test the parser independently of OpenBabel typing.

## 11. MolOptima deployment implications

| Concern | RDKit-only | RDKit + OpenBabel |
|---|---|---|
| Scientific content of released six features | Preserves coordinates, released elemental radii, exact masks/kernels, and six statistics | Same only if typing, coordinates, and row mapping survive conversion |
| Dependency burden | Uses the existing molecular toolkit | Adds a second chemistry toolkit and BioPandas/MOL2 boundary |
| Windows/desktop packaging | Existing RDKit deployment path; fewer native binaries | Additional native binary/runtime and Conda/channel compatibility work |
| Atom-order reliability | Direct construction in the same RDKit heavy order used by Chemprop | Must prove and possibly reverse converter reordering |
| Coordinate reproducibility | Uses selected conformer coordinates directly | MOL2 precision/rounding can perturb distances and high-power kernels |
| Chemistry perception | Atomic number only, which is all the public six-feature path needs | SYBYL aromatic/amide/charge perception is computed but numerically discarded after radius lookup |
| Maintenance | One identity authority and an explicit small radius contract | Cross-tool versions, format parsing, mapping, and typing fixtures must be maintained |
| Failure surface | Unsupported elements and geometry failures can be explicit | Adds conversion, type-vocabulary, atom-name/order, charge, bond, and parser failures |

RDKit-only is recommended because it is scientifically equivalent on the released supported
domain, not merely because it is easier to deploy.

## 12. Final answers

1. **Is OpenBabel required for the released BBBP six-feature GGL?** No.
2. **Is MOL2 required mathematically?** No.
3. **Is MOL2 only a legacy input format for the released extractor?** Yes. It supplies coordinates,
   an over-specific route to elemental radii, and row order.
4. **Are SYBYL atom types used numerically?** The strings are used as lookup keys, but the
   fine-grained SYBYL subtype is numerically irrelevant. Only element-specific radii survive into
   the calculation.
5. **Can RDKit atomic identity replace MOL2 typing/filtering?** Yes. Atomic number can select the
   exact released element radius and exclude atomic-number-1 hydrogens. Unsupported elements should
   fail explicitly.
6. **Can direct RDKit heavy-atom ordering eliminate atom-remapping risk?** Yes. It makes GGL and
   Chemprop row identity the same by construction, subject to explicit index-preservation checks.
7. **Can RDKit-only reasonably be called a reproduction?** Yes: a reproduction of the public
   pooled six-feature GMC representation. No: it is not the unreleased colored WCS implementation
   or an exact reproduction of published geometries/results.
8. **What geometry deviation must be disclosed?** The authors' primary OMEGA geometry generation,
   lowest-energy selection details, versions, seeds, and structures are unavailable; their stated
   fallback used ETKDGv2 -> PDB -> OpenBabel MOL2. Deterministic ETKDGv3 plus MMFF94s/documented UFF
   fallback is a transparent replacement and can produce different conformers and therefore
   different GGL values.

Additional reproducibility caveats remain: the public source's pooled six columns do not implement
the paper's colored channels; the public mask differs from the paper's stated non-covalent rule;
and kernel-selection provenance is insufficient for exact published-result reconstruction. These
do not create a need for SYBYL typing or OpenBabel.

## 13. Selected next task

**Implement deterministic RDKit geometry plus direct six-feature GGL calculation on five synthetic
molecules only.**

The task should independently implement the literal public calculation, use the explicit released
element-radius mapping, preserve original RDKit heavy order, test subtype invariance and mask
boundaries, and implement no BBB preprocessing, model, or training code.
