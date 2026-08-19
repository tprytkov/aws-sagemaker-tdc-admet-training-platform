# GMC-MPNN Phase-1 parent-fragment policy

The BBB_Martins PyTDC-0.3.9 development-only audit found 96 disconnected structures among
1,757 train and validation rows. Applying the candidate parent rule produced seven collision
groups of size at most two, no label-conflict groups, no new train-validation exact-parent
overlap, and no new train-validation Murcko-scaffold overlap. The locked test was not accessed.

For GMC-MPNN Phase 1 only, connected structures remain unchanged. Ordinary disconnected
structures use the fragment with the largest heavy-atom count for geometry; ties are resolved by
ascending canonical isomeric SMILES. The original source structure, source-row identity, split,
and removed-fragment provenance remain unchanged and separate from the geometry representation.
Parent collisions are not deduplicated by this policy.

The exact molecule IDs `eqvalan` and `sultamicillin` are frozen policy exclusions and are not
automatically reduced to a largest fragment. This policy controls only which representation enters
GMC-MPNN geometry; it does not modify the BBB source CSVs or the geometry/GGL algorithms.
