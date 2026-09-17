# Invariant / Equivariant Crystal Generation Survey

The project goal is experimental crystal-structure analysis. The key model
requirement is to avoid arbitrary global-frame dependence while supporting:

```text
PXRD only
PXRD + element set
PXRD + XRF
PXRD + neutron
PXRD + neutron + XRF
PXRD + neutron + XRF + XPS-valence
PXRD + exact formula
lattice + conditions
conditions without lattice
```

## Main Families

### Periodic E(3)-Equivariant Diffusion

Representative model: DiffCSP.

This family denoises lattice and fractional coordinates with periodic E(3)
equivariant graph networks. It directly addresses translation, rotation,
permutation, and periodic-boundary symmetries.

Project value:

- stable baseline for CSP and conditional reconstruction;
- MIT licensed;
- close to existing `StructureMatcher` top-k evaluation.

Limitation:

- diffusion sampling can be slower than flow matching;
- multimodal spectrum conditioning is not the default task and must be added.

### Riemannian / Crystal Flow Matching

Representative models: FlowMM, CrystalFlow.

This family learns vector fields on manifolds appropriate for atom types,
lattice variables, and periodic fractional coordinates. It is a strong fit for
the new mainline because flow matching is efficient and conditionable.

Project value:

- newer than DiffCSP and more aligned with current generative modeling trends;
- natural path to condition on PXRD, neutron, XRF, XPS, and lattice embeddings;
- promising for reward-guided candidate refinement because sampling is faster.

Limitation:

- FlowMM is CC-BY-NC, so it is appropriate for academic/non-commercial work;
- CrystalFlow is newer and needs deeper engineering validation before becoming
  the only dependency.

### Space Group / Wyckoff Equivariant Generation

Representative models: SGEquiDiff, WyckoffDiff.

These models encode crystallographic symmetry more explicitly through space
groups, Wyckoff positions, and asymmetric units.

Project value:

- strong novelty for known-space-group or indexed-lattice settings;
- useful later for reducing search space and improving validity.

Limitation:

- harder first integration because our input may not always include space group;
- the immediate task needs spectrum-conditioned structure solving, not only
  symmetry-prototype generation.

### Invariant Internal Representations

Examples: distance/contact graphs, pair distributions, local environments,
SLICES-like strings.

These representations can remove global rotation/translation dependence, but
they often lose phase/coordinate information or require a separate reconstruction
step. For our target, they are better used as auxiliary rewards or conditions
than as the only generator representation.

## Selected Direction

Use FlowMM as the primary mainline and DiffCSP as the stable baseline:

```text
Primary:   FlowMM-style Riemannian flow matching
Baseline:  DiffCSP-style periodic E(3)-equivariant diffusion
Later:     space-group/Wyckoff constraints for indexed-lattice cases
```

The core innovation should be:

```text
multimodal spectrum-conditioned crystal flow matching
```

Concretely, add condition encoders for PXRD, neutron diffraction, XRF mass
fractions, XPS-style valence assignments, exact formula, and optional lattice.
Then compare every condition set against the archived Uni-3DAR baseline.
