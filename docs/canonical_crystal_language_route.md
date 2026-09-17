# Canonical CIF-Like Crystal Language Route

This note defines the first language-model route for experimental crystal
structure solving. The goal is to test whether a Transformer-style
autoregressive model can map multimodal experimental conditions to a canonical
CIF-like crystal sequence.

This route is deliberately simpler than Mat2Seq-style invariant tokenization.
It should be treated as the first version for validating the sequence-model
hypothesis before adding stronger canonicalization or symmetry-reduced
representations.

## Scope

This language-model route is developed in parallel with the FlowMM route. It
must not replace, block, or modify the FlowMM/DiffCSP crystal-native track.
Shared data preparation and evaluation utilities are acceptable, but model code,
configs, checkpoints, and experiment outputs should remain separate.

The first training stage should use only the MP+WBM multimodal dataset rather
than the full 10M+ unified corpus. MP+WBM is the small validation track for
testing whether autoregressive crystal language modeling is viable before
scaling to the full multi-source dataset.

## Objective

Train an autoregressive crystal language model for:

```text
experimental conditions -> canonical CIF-like crystal tokens
```

The conditions follow the project ablation ladder:

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

The output is a normalized structure sequence that can be parsed into a
`pymatgen.Structure`, then scored with `StructureMatcher` and simulated
PXRD/neutron rewards.

## Sequence Template

Each training example is a single sequence with a condition prefix and a target
crystal suffix:

```text
<BOS>
<TASK> solve_structure
<COND_START>
...
<COND_END>
<CRYSTAL_START>
...
<CRYSTAL_END>
<EOS>
```

During training, loss is applied primarily to the crystal suffix. The condition
prefix is provided as context. During inference, the model receives tokens up to
`<CRYSTAL_START>` and autoregressively generates the crystal suffix.

## Condition Blocks

### PXRD

Use a compact peak-token representation for the first implementation:

```text
<PXRD_START>
<PXRD_META> radiation CuKa wavelength 1.5406 two_theta_min 5 two_theta_max 90
<PXRD_PEAK> two_theta 27.320 intensity 1.000 width 0.120
<PXRD_PEAK> two_theta 31.680 intensity 0.520 width 0.140
<PXRD_PEAK> two_theta 45.420 intensity 0.330 width 0.160
<PXRD_END>
```

For synthetic datasets, peak extraction can be deterministic. Later versions can
replace or augment this with a learned 1D spectrum encoder prefix.

### Neutron Diffraction

Use the same peak grammar, but keep the modality separate because neutron
intensities encode different scattering contrast:

```text
<NEUTRON_START>
<NEUTRON_META> wavelength 1.5406 q_min 0.5 q_max 12.0
<NEUTRON_PEAK> q 1.930 intensity 0.410 width 0.080
<NEUTRON_PEAK> q 2.230 intensity 0.880 width 0.090
<NEUTRON_END>
```

The model should learn that PXRD and neutron peaks constrain the same lattice
and atomic arrangement through different scattering factors.

### Element Set

Element-set conditioning is composition-light and does not specify exact ratios:

```text
<ELEMENT_SET> Na Cl
```

This is useful for evaluating whether spectra plus possible elements can solve
the structure without exact stoichiometry.

### XRF

XRF is represented as a soft composition constraint:

```text
<XRF_START>
<XRF_MASS_FRAC> Na 0.393
<XRF_MASS_FRAC> Cl 0.607
<XRF_END>
```

This should constrain element ratios without requiring exact integer formula
knowledge.

### XPS Valence

XPS-style valence is represented as likely oxidation-state evidence:

```text
<XPS_VALENCE_START>
<VALENCE> Na 1.0 confidence 0.90
<VALENCE> Cl -1.0 confidence 0.88
<XPS_VALENCE_END>
```

The first version can use simulated or database-derived valence assignments.
Later versions can support multiple candidate valences per element.

### Exact Formula

Exact formula is the oracle-like composition condition:

```text
<FORMULA> Na 1 Cl 1
```

This ablation tests how close the model can get when the strongest composition
constraint is supplied.

### Optional Lattice

Known-lattice settings include a lattice block in the condition prefix:

```text
<KNOWN_LATTICE>
<LATTICE> a 5.640 b 5.640 c 5.640 alpha 90.000 beta 90.000 gamma 90.000
```

In lattice-conditioned training, the output should still include the lattice so
that the generated structure is self-contained, but lattice-token loss can be
masked or down-weighted if desired.

## Output Crystal Blocks

The first output format should be canonical CIF-like rather than raw CIF. It is
easier to parse, easier to constrain, and less sensitive to metadata variation.

```text
<CRYSTAL_START>
<FORMULA> Na 1 Cl 1
<N_SITES> 2
<SPACE_GROUP> number 225 symbol Fm-3m
<LATTICE> a 5.640 b 5.640 c 5.640 alpha 90.000 beta 90.000 gamma 90.000
<ATOM_SITES_START>
<ATOM> Na x 0.0000 y 0.0000 z 0.0000 occ 1.000
<ATOM> Cl x 0.5000 y 0.5000 z 0.5000 occ 1.000
<ATOM_SITES_END>
<CRYSTAL_END>
```

Optional auxiliary blocks can be added after the minimal format is stable:

```text
<LOCAL_ENV_START>
<ENV> Na coord 6 neighbor Cl poly oct mean_dist 2.820 distortion low
<ENV> Cl coord 6 neighbor Na poly oct mean_dist 2.820 distortion low
<LOCAL_ENV_END>
```

These local-geometry tokens should be auxiliary targets, not required fields in
the first parser.

## Canonicalization Rules

The first version should use simple deterministic normalization:

1. Convert the source structure to a standard conventional cell when reliable;
   otherwise fall back to a primitive cell.
2. Wrap fractional coordinates into `[0, 1)`.
3. Round lattice parameters and fractional coordinates to fixed precision.
4. Sort atoms by element, then fractional `x`, `y`, `z`, then occupancy.
5. Keep one canonical spelling for space-group symbols.
6. Emit all fields in a fixed block order.

This does not guarantee full Mat2Seq-style invariance, but it reduces target
entropy enough to test the language-model route.

## Ablation Templates

All ablations share the same output grammar. They differ only in the condition
prefix.

### A0: PXRD Only

```text
<TASK> solve_structure
<COND_START>
<PXRD_START> ... <PXRD_END>
<COND_END>
<CRYSTAL_START> ...
```

### A1: PXRD + Element Set

```text
<COND_START>
<PXRD_START> ... <PXRD_END>
<ELEMENT_SET> Na Cl
<COND_END>
```

### A2: PXRD + XRF

```text
<COND_START>
<PXRD_START> ... <PXRD_END>
<XRF_START> ... <XRF_END>
<COND_END>
```

### A3: PXRD + Neutron

```text
<COND_START>
<PXRD_START> ... <PXRD_END>
<NEUTRON_START> ... <NEUTRON_END>
<COND_END>
```

### A4: PXRD + Neutron + XRF

```text
<COND_START>
<PXRD_START> ... <PXRD_END>
<NEUTRON_START> ... <NEUTRON_END>
<XRF_START> ... <XRF_END>
<COND_END>
```

### A5: PXRD + Neutron + XRF + XPS-Valence

```text
<COND_START>
<PXRD_START> ... <PXRD_END>
<NEUTRON_START> ... <NEUTRON_END>
<XRF_START> ... <XRF_END>
<XPS_VALENCE_START> ... <XPS_VALENCE_END>
<COND_END>
```

### Oracle: PXRD + Exact Formula

```text
<COND_START>
<PXRD_START> ... <PXRD_END>
<FORMULA> Na 1 Cl 1
<COND_END>
```

### Lattice + Conditions

The lattice block can be added to any condition set:

```text
<COND_START>
<KNOWN_LATTICE>
<LATTICE> a 5.640 b 5.640 c 5.640 alpha 90.000 beta 90.000 gamma 90.000
<PXRD_START> ... <PXRD_END>
<XRF_START> ... <XRF_END>
<COND_END>
```

### Conditions Without Lattice

The same condition set without `<KNOWN_LATTICE>` is the lattice-free comparison.
The model must generate the lattice in the crystal suffix.

## Training Tasks

Start with three supervised tasks:

```text
T1: conditions -> full crystal tokens
T2: conditions + partial crystal prefix -> remaining crystal tokens
T3: conditions + corrupted crystal tokens -> corrected crystal tokens
```

T1 tests direct structure solving. T2 supports staged decoding, such as lattice
first and atom sites second. T3 prepares the model for candidate refinement and
reward-guided correction.

## Evaluation

Use the same top-k structure-analysis metrics as the crystal-native route:

```text
valid parse rate
composition/formula consistency
StructureMatcher top-1 and top-k match rate
matched RMS distance
PXRD profile similarity after re-simulation
neutron profile similarity when neutron input is present
XRF mass-fraction consistency
XPS-valence consistency
duplicate-adjusted top-k diversity
```

For language-model-specific reporting, also track:

```text
syntax-valid sequence rate
numeric-token validity
lattice-validity rate
atom-site count consistency
round-trip Structure parse success
```

## Implementation Notes

The first implementation should avoid raw CIF metadata and output only the
minimal canonical blocks needed to build `pymatgen.Structure`.

Recommended parser contract:

```text
tokens -> formula, n_sites, optional space_group, lattice, atom_sites
```

Recommended generator contract:

```text
record -> one sequence per ablation condition set
```

This makes it possible to build all ablation datasets from the existing
MP+WBM/unified multimodal records without changing the underlying source data.
