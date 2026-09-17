# Crystal-Native Structure Generation Route

This note turns the model-route decision into an executable project track. The
project keeps Uni-3DAR as the fixed baseline and moves the main structure
generator toward crystal-native equivariant methods.

## Decision

Do not continue large architectural work to retrofit Uni-3DAR with translation
and rotation invariance. Uni-3DAR should remain the baseline for data, condition
encoding, and ablation reporting.

The mainline generator should use a representation that is native to periodic
crystals:

- lattice matrix or lattice parameters;
- fractional coordinates modulo the unit cell;
- atom types with permutation-invariant or permutation-equivariant treatment;
- periodic boundary conditions in neighborhoods and losses;
- optional lattice conditioning without forcing a global voxel/octree frame.

This avoids relying on canonicalization such as center-of-geometry or
center-of-mass alignment, which is not a physically natural invariant for
periodic crystals.

## Frozen Uni-3DAR Baseline

Current baseline code paths:

- training: `baselines/uni3dar/scripts/train_official_unified.sh`
- SLURM wrapper: `baselines/uni3dar/scripts/submit_train_official_unified.sbatch`
- inference ablation: `baselines/uni3dar/scripts/infer_mp_wbm_ablation.sh`
- ablation wrapper: `baselines/uni3dar/scripts/submit_infer_mp_wbm_ablation.sbatch`
- condition builder: `baselines/uni3dar/uni3dar/data/crystal_data_utils.py`
- crystal grid conversion: `baselines/uni3dar/uni3dar/data/crystal_grid_dataset.py`

Baseline checkpoint used by the MP+WBM ablation:

```text
/home/tkh/ckps/wuji_yl_official_unified_20260521_171452/ckpt/checkpoint_best.pt
```

The baseline was trained with PXRD, neutron diffraction, element set, exact
formula ratio, XRF mass fraction, and XPS-style valence conditions. Each
condition is independently dropped during training.

Current MP+WBM ablation snapshot (`LEN_DATASET=50`, `NUM_SAMPLES=20`,
reported from rank 0, batch size 2):

| Condition | Input conditions | Top-1 match | Top-10 match | Matched Top-1 RMSE | Matched Top-10 RMSE |
|---|---|---:|---:|---:|---:|
| A0 | PXRD | 4 / 50 | 4 / 50 | 0.1964 | 0.1956 |
| A1 | PXRD + element set | 9 / 50 | 12 / 50 | 0.0828 | 0.1331 |
| A2 | PXRD + XRF | 7 / 50 | 7 / 50 | 0.1575 | 0.1468 |
| A3 | PXRD + neutron | 4 / 50 | 4 / 50 | 0.1862 | 0.1834 |
| A4 | PXRD + neutron + XRF | 6 / 50 | 8 / 50 | 0.1822 | 0.1840 |
| A5 | PXRD + neutron + XRF + XPS valence | 13 / 50 | 15 / 50 | 0.1778 | 0.1443 |
| Oracle | PXRD + exact formula ratio | 4 / 50 | 5 / 50 | 0.0336 | 0.1009 |

Interpretation:

- A5 is the strongest current non-oracle setting, but absolute match rate is
  still low.
- Exact formula is not reliably exploited by the current representation.
- The baseline is useful for measuring the value of the multi-modal conditions,
  but it should not define the future structure representation.

## Candidate Backbones

Evaluate two crystal-native families first.

Current default mainline:

1. Start implementation with FlowMM-style crystal flow matching.
2. Keep DiffCSP-style periodic diffusion as the stable baseline challenger.
3. Add space-group / Wyckoff constraints later for indexed-lattice cases.

### Candidate A: FlowMM / crystal flow-matching style generator

Why it is the mainline candidate:

- flow matching is a strong modern alternative to diffusion;
- continuous vector fields are natural for multi-modal conditioning;
- sampling should be more efficient for top-k candidate generation and reward
  reranking;
- it directly models lattice, fractional coordinates, atom types, and periodic
  symmetries instead of a fixed global grid.

Expected integration cost:

- medium;
- requires a JSONL dataset adapter plus condition encoders for PXRD/neutron/XRF/XPS;
- upstream license is CC-BY-NC, so this route is for academic/non-commercial research.

Primary role in bake-off:

- main candidate for `conditions + optional lattice -> ranked structures`.

### Candidate B: DiffCSP-style periodic diffusion

Why it is the stable baseline challenger:

- established target for crystal structure prediction;
- operates on lattice, fractional coordinates, and atom types rather than a
  fixed global grid;
- existing metric practice already aligns with `StructureMatcher` and top-k
  CSP evaluation;
- MIT licensed.

Expected integration cost:

- low-to-medium;
- requires a conditioning adapter for spectra and composition;
- can start with formula/PXRD/lattice conditions before adding neutron and XPS.

Primary role in bake-off:

- stable diffusion comparison against FlowMM on the same exported JSONL splits.

## Bake-Off Protocol

Use the same MP+WBM splits that produced the Uni-3DAR ablation before expanding
to the full unified multi-source dataset.

Minimum condition sets:

| Name | Conditions | Purpose |
|---|---|---|
| B0 | composition only | sanity check and unconditional crystal prior |
| B1 | PXRD + composition | first real structure-analysis setting |
| B2 | PXRD + composition + lattice | tests whether known lattice helps |
| B3 | PXRD + neutron + composition + lattice | tests diffraction complementarity |
| B4 | PXRD + neutron + XRF + XPS valence + lattice | full multi-modal setting |

Primary metrics:

- valid structure rate after constructing a `pymatgen.Structure`;
- composition/formula consistency;
- `StructureMatcher` top-1 and top-k match rate;
- matched RMS distance;
- PXRD cosine or weighted profile similarity after re-simulation;
- neutron pattern similarity when neutron input is present;
- duplicate-adjusted top-k diversity.

The key gate is that lattice-conditioned runs must improve over lattice-free
runs on physically relevant metrics. If known lattice hurts the new backbone,
the candidate should not become the mainline.

## Minimal Data Adapter

The adapter target is JSONL so that each candidate backbone can implement its
own thin dataset class without depending on Uni-Core internals.

One record should contain:

```json
{
  "id": "mp:mp-1234",
  "source": "mp",
  "source_id": "mp-1234",
  "split": "train",
  "atom_types": ["Na", "Cl"],
  "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
  "lattice_matrix": [[...], [...], [...]],
  "formula": "NaCl",
  "n_sites": 2,
  "space_group": 225,
  "conditions": {
    "pxrd_x": [...],
    "pxrd_y": [...],
    "neutron_x": [...],
    "neutron_y": [...],
    "xrf_mass_frac": {"Na": 0.3934, "Cl": 0.6066},
    "valence_assignments": {"Na": 1.0, "Cl": -1.0}
  },
  "properties": {
    "formation_energy_per_atom": -1.0,
    "e_above_hull": 0.0,
    "band_gap": 5.0
  }
}
```

Use `data_prep/official_multisource/scripts/export_crystal_native_jsonl.py` to
create the first bake-off export.

Example:

```bash
python data_prep/official_multisource/scripts/export_crystal_native_jsonl.py \
  --data-dir data_prep/outputs/official_unified_multimodal \
  --out-dir data_prep/outputs/crystal_native_unified \
  --max-records-per-split 1000
```

## Mainline Selection Rule

Until the bake-off is run, the selected engineering track is:

```text
Primary:   FlowMM-style crystal flow matching
Challenger: DiffCSP-style periodic diffusion
Baseline:  frozen Uni-3DAR MP+WBM multimodal checkpoint
```

Promote the new backbone only if it beats the frozen Uni-3DAR baseline under the
same data split and sampling budget.

Minimum promotion criteria:

- better Top-1 and Top-10 `StructureMatcher` match rate than A5;
- lattice-conditioned performance is better than lattice-free performance;
- no large drop in valid structure rate;
- straightforward path to add neutron, XRF, and XPS conditions;
- generated candidates can be passed to the existing physics reward stack.

After promotion, the training route becomes:

1. crystal-native supervised pretraining on MP+WBM, then unified multi-source
   LMDB exports;
2. multi-modal structure-analysis SFT;
3. physics reward reranking;
4. reward-weighted SFT;
5. optional RL/GRPO-style fine-tuning only after reranking is stable.
