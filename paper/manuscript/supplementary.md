<!--
DRAFT HEADER — remove before submission.

This SI round (2026-09-13)
  Wrote Supplementary Notes 1–16, Supplementary Figs. 1–14, Supplementary Tables 1–17.
  Main text and figure captions cite these items by number.

Included (from paper_figures/si_supplementary)
  fig_s13 token length → Supplementary Fig. 1
  fig_s2 PXRD vs ND → Supplementary Fig. 2
  fig_s6 governance → Supplementary Fig. 3
  fig_s1 ladder with XPS → Supplementary Fig. 4
  fig_s5 XPS increment → Supplementary Fig. 5
  fig_s3 ND by system → Supplementary Fig. 6
  fig_s4 size bins → Supplementary Fig. 7
  fig_s14 ND rerank weight → Supplementary Fig. 8
  fig_s11_clean_vs_noisy → Supplementary Fig. 9
  fig_s15 runtime → Supplementary Fig. 10
  fig_s16 MP-20 leakage → Supplementary Fig. 11
  fig_s12 collaborator ranks → Supplementary Fig. 12
  fig_s7 cosine QC (construction, not accuracy) → Supplementary Fig. 13
  fig_s8 filter counts (construction) → Supplementary Fig. 14

Deliberately not used as results
  fig_s9 / fig_s9b / fig_s10_selective_prediction / retrieval_baseline_public /
  profile_residual_public / realpxrd_protocol_match.json accuracies:
  these are the retired public-337 numbers (including 8.6%). Accuracies reserved.
  fig_s10_mp20_protocol.svg plots our first-500 83.2% on the same axis as PXRDGen
  full 9046 — not used. MP-20 literature and our first-500 are two tables, never one plot.

No 337 refine job is promised.
-->

# Supplementary Information

**Complementary X-ray and neutron probes for crystal-structure solution in one autoregressive language model**

This file accompanies the main text. Supplementary Notes give protocols and negative results; Supplementary Figures reuse the drawings in `paper_figures/si_supplementary/outputs` under a new numbering that matches the Notes; Supplementary Tables collect rates that do not belong on a main figure. A public experimental PXRD hold-out is described only as a recoding protocol (Supplementary Note 15). No accuracy from that collection is reported here.

---

## Supplementary Note 1. Evaluation protocol and metrics

Every structure-match rate in the main text is a reranked Top-*k* (`topk_rerank`) unless a caption says otherwise. For each query we sample *N*_cand = 100 sequences, parse them to `pymatgen` `Structure` objects, drop parse failures, collapse near-duplicates, and order the remainder by the physical reranker of Supplementary Note 8. Top-*k* is the fraction of queries for which at least one of the first *k* reranked candidates matches the reference under `StructureMatcher` (`ltol=0.3`, `stol=0.5`, `angle_tol=10`, `primitive_cell=True`). Composition recovery is an exact reduced-formula match after parse; it is not a structure match.

Two lattice settings are reported throughout. **Lattice-free**: the six cell parameters are not in the prompt; the decoder emits them. **Known lattice**: the cell is injected as condition tokens and is not sampled. Formula-constrained decoding is used only when the exact formula is in the prompt.

The **formula reference** is PXRD plus the exact stoichiometry. It is an oracle input, not a theoretical upper bound: a spectroscopic prefix can outrank it on physical consistency, which is why known-lattice Multi Top-1 (90.4%) sits 0.8 points above the formula reference (89.5%) on the full test set (main Fig. 3f).

Condition names used in the paper are listed in Supplementary Table 1. XPS-style valence tokens are in the training mix but are omitted from every main figure (Supplementary Note 5).

**Supplementary Table 1 | Condition names.**

| Name in text | Prefixes | Formula-constrained decoding |
|---|---|---|
| PXRD | PXRD only | no |
| PXRD+ND | PXRD and neutron powder diffraction | no |
| PXRD+XRF | PXRD and exact elemental mass fractions | no |
| Multi (PXRD+ND+XRF) | PXRD, neutron, mass fractions | no |
| Formula reference | PXRD and exact formula | yes |
| PXRD+ND+XRF+XPS | Multi plus per-element valence tokens | no; SI only |

---

## Supplementary Note 2. Crystal language and token length

The v2 grammar is CIF-like and deterministic: `<CRYSTAL_START> <FORMULA> … <N_SITES> … <SPACE_GROUP> … <LATTICE> … <ATOM_SITES_START> … <CRYSTAL_END>`. Fractional coordinates occupy 1,000 bins (`<C000>`–`<C999>`), lattice lengths 0.05 Å bins from 1–50 Å, lattice angles 0.5° bins from 30–150°, and space groups `<SG001>`–`<SG230>`. Sites are sorted by element then coordinates. The paper tokenizer has 2,580 tokens. PXRD and neutron patterns are not in this stream (Supplementary Note 3).

Supplementary Fig. 1 shows v2 target-token length versus site count on the complete simulated test set (*N* = 33,944). The median length is 71 tokens, the 90th percentile 142, and the maximum 671, all inside the 1,024-token context. Autoregressive sampling of *N* = 100 candidates per query is therefore a length-feasible operating point, not a truncated one (see also Supplementary Note 10).

**Supplementary Figure 1 | Crystal-language length on the simulated test set.**
V2 target tokens versus *n*_sites. Median = 71. Source: `fig_s13_v2_token_length.csv`. Repository file: `fig_s13_v2_token_length.svg`.

---

## Supplementary Note 3. Spectra prefixes and paired simulation physics

PXRD and neutron powder patterns are converted to (*d*, intensity) peak lists (maximum 128 peaks) and encoded by a three-layer peak-set transformer into 16 prefix slots at `<PXRD_EMB>` and `<ND_EMB>`. Encoding *d* rather than a fixed 2θ raster removes the diffractometer wavelength from the prefix geometry. Missing modalities zero the corresponding slots.

All simulated patterns in the Tier-A corpus use Cu Kα (λ = 1.54184 Å) over 5–80° 2θ. Neutron patterns are generated with `pymatgen` `NDCalculator` on the same angular window so that the two prefixes are paired, not independently rescaled. X-ray intensities weight electron density (monotonic in Z); neutron intensities weight coherent scattering lengths, which are non-monotonic in Z and remain finite for H and Li. Supplementary Fig. 2 is a schematic of that contrast; quantitative ND increments are in main Fig. 4 and Supplementary Tables 6 and 7.

XRF in every main-text ladder is a token block of **exact elemental mass fractions** to three decimals. It is not a measured energy-dispersive spectrum. XPS-style tokens are a per-element valence proxy, not an XP spectrum (Supplementary Note 5).

**Supplementary Figure 2 | Complementary PXRD and neutron contrast.**
Schematic traces. Not data. Repository file: `fig_s2_pxrd_vs_nd.svg`.

---

## Supplementary Note 4. Governance of the multimodal corpus

The scientific product of the data work is a leakage-audited, uniformly simulated multimodal split, not a new source database. Supplementary Fig. 3 restates the funnel of main Fig. 2a as counts (Supplementary Table 2). Unified records: 10,207,687. Tier-A quality filter: 4,127,590 (40.4% of unified). Tight structure-level deduplication: 3,524,185 (34.5% of unified). Frozen split: train 3,456,202 / valid 34,039 / test 33,944. Deduplication priority was Materials Project > WBM > OQMD > Alexandria > AFLOW. After deduplication the kept mix is Alexandria 62.6% (2,206,199), AFLOW 15.3% (539,449), OQMD 10.8% (379,297), WBM 7.1% (250,183), Materials Project 4.2% (149,057) (Supplementary Table 3).

Train / valid / test assignments were frozen after a two-stage leakage audit: coarse composition–lattice buckets, then `StructureMatcher` confirmation. The test scan of 33,944 records is complete (no sampling). Test cells: median 10 sites, mean 12.8, 95th percentile 36, range 1–128; median three distinct elements; 85 elements present. Crystal-system counts from space-group tokens are in Supplementary Table 4.

External MP-20 leakage relative to a RealPXRD-style training pool is a different question and is isolated in Supplementary Note 11. It is not mixed into Fig. 2 or Fig. 3.

**Supplementary Figure 3 | Governance funnel counts.**
Repository file: `fig_s6_governance_counts.svg`.

**Supplementary Table 2 | Frozen split.**

| Split | Records | Fraction of 3,524,185 |
|---|---:|---:|
| Train | 3,456,202 | 98.07% |
| Valid | 34,039 | 0.97% |
| Test | 33,944 | 0.96% |

**Supplementary Table 3 | Source mix after tight deduplication (*N* = 3,524,185).**
Priority during deduplication was Materials Project > WBM > OQMD > Alexandria > AFLOW.

| Source | Records | Share |
|---|---:|---:|
| Alexandria | 2,206,199 | 62.6% |
| AFLOW | 539,449 | 15.3% |
| OQMD | 379,297 | 10.8% |
| WBM | 250,183 | 7.1% |
| Materials Project | 149,057 | 4.2% |

**Supplementary Table 4 | Crystal-system counts on the complete simulated test set (*N* = 33,944).**
Mapped from space-group tokens. These *n* are not the aligned 10,000-structure subset of Supplementary Table 7.

| System | Space groups | Records | Share |
|---|---|---:|---:|
| Triclinic | 1–2 | 1,502 | 4.4% |
| Monoclinic | 3–15 | 8,104 | 23.9% |
| Orthorhombic | 16–74 | 7,676 | 22.6% |
| Tetragonal | 75–142 | 7,187 | 21.2% |
| Trigonal | 143–167 | 5,325 | 15.7% |
| Hexagonal | 168–194 | 2,096 | 6.2% |
| Cubic | 195–230 | 2,054 | 6.1% |

---

## Supplementary Note 5. XPS-style valence is not a main-text probe

XPS-style tokens store a per-element oxidation-state guess, not a measured XP spectrum. They are in the mix so that the decoder sees the grammar, and they are omitted from main Figs. 3–5 because the increment on the full simulated test is near zero.

On *N* = 33,944, lattice-free Top-1 is 84.0% for Multi and 84.0% after adding XPS (+0.03 points). Known-lattice Top-1 is 90.4% for Multi and 90.3% after XPS (−0.06 points) (Supplementary Figs. 4, 5; Supplementary Table 5). The faded XPS row in main Fig. 6e is the same negative result under synthetic artefacts: on a frozen noisy lattice-free pool, Multi+XPS Top-1 is 76.8% before Rietveld-lite and 79.6% after, below Multi (78.6% / 82.2%). We treat XPS as a negative control, not as an unused selling point.

**Supplementary Figure 4 | Full simulated ladder including the XPS rung.**
Lattice-free and known-lattice Top-1, *N* = 33,944, *N*_cand = 100. XPS is the fifth bar. Repository file: `fig_s1_simulated_ladder_top1.svg`.

**Supplementary Figure 5 | XPS increment on the full simulated test.**
Multi versus Multi+XPS. Repository file: `fig_s5_xps_increment.svg`.

**Supplementary Table 5 | XPS increment (reranked Top-1, *N* = 33,944).**

| Lattice | Multi | +XPS | Δ (points) |
|---|---:|---:|---:|
| Lattice-free | 84.0% | 84.0% | +0.03 |
| Known lattice | 90.4% | 90.3% | −0.06 |

**Supplementary Table 6 | Full simulated ladder, reranked Top-*k* (*N* = 33,944, *N*_cand = 100).**
Percentages rounded as in the main text. Source: `figure_3_source_data.csv`. XPS is not in this table (see Supplementary Table 5).

*Lattice-free*

| Condition | Comp. rec. | T1 | T5 | T10 | T20 | T50 |
|---|---:|---:|---:|---:|---:|---:|
| PXRD | 24.8 | 3.8 | 8.8 | 11.8 | 15.2 | 20.3 |
| PXRD+ND | 31.6 | 7.8 | 14.3 | 17.9 | 21.5 | 26.6 |
| PXRD+XRF | 98.9 | 83.2 | 88.7 | 90.2 | 91.4 | 92.3 |
| Multi | 98.9 | 84.0 | 89.0 | 90.4 | 91.5 | 92.3 |
| Formula reference | 100 | 85.9 | 90.7 | 92.1 | 93.0 | 93.7 |

*Known lattice*

| Condition | Comp. rec. | T1 | T5 | T10 | T20 | T50 |
|---|---:|---:|---:|---:|---:|---:|
| PXRD | 32.1 | 7.1 | 16.3 | 20.6 | 24.7 | 28.7 |
| PXRD+ND | 39.4 | 15.5 | 26.1 | 30.0 | 33.2 | 35.9 |
| PXRD+XRF | 99.8 | 89.4 | 92.4 | 93.1 | 93.7 | 94.1 |
| Multi | 99.8 | 90.4 | 92.6 | 93.2 | 93.6 | 94.0 |
| Formula reference | 100 | 89.5 | 92.5 | 93.3 | 93.9 | 94.3 |

---

## Supplementary Note 6. Neutron gains by crystal system

Main Fig. 4b is the claim: paired bootstrap intervals on an aligned *N* = 10,000 subset. Crystal-system cells in Fig. 4c–f are descriptive point estimates; the source artefact does not provide subgroup confidence intervals. Supplementary Table 7 and Supplementary Fig. 6 list every system so that a reader can see the hexagonal/cubic concentration of PXRD→+ND gain and the near-zero, sign-changing ND-given-XRF cells. The *n* in Supplementary Table 7 is that 10k subset, not the full-test counts of Supplementary Table 4.

**Supplementary Figure 6 | Neutron Top-1 gain by crystal system.**
Repository file: `fig_s3_nd_gain_by_system.svg`. Values from `ladder_stats.json` via `crystal_system_neutron_effects.csv`.

**Supplementary Table 7 | Crystal-system neutron gains (aligned 10k subset; descriptive).**
Δ is Top-1(PXRD+ND) − Top-1(PXRD), in percentage points. ND|XRF is Top-1(Multi) − Top-1(PXRD+XRF).

| System | *n* | Free Δ ND | Free Δ ND\|XRF | Known Δ ND | Known Δ ND\|XRF |
|---|---:|---:|---:|---:|---:|
| Triclinic | 129 | 0.0 | −0.8 | +5.4 | +2.3 |
| Monoclinic | 2,049 | +1.6 | +0.3 | +6.6 | +1.0 |
| Orthorhombic | 2,459 | +4.1 | +0.1 | +7.9 | +0.6 |
| Tetragonal | 3,125 | +5.1 | +0.6 | +9.7 | +0.8 |
| Trigonal | 1,117 | +5.6 | +0.8 | +9.0 | +0.5 |
| Hexagonal | 499 | +10.0 | +1.2 | +15.4 | +0.4 |
| Cubic | 622 | +9.3 | −0.8 | +11.9 | −0.3 |

Aggregate ND given XRF (paired *N* = 10,000): lattice-free +0.37 points (95% CI −0.07 to +0.82, crosses zero); known lattice +0.68 points (0.40–0.97).

---

## Supplementary Note 7. Size bins on the full test set

Main Fig. 5a,b already report the full-test split. Supplementary Fig. 7 is the same comparison in the original SI layout. Counts: ≤20 sites *n* = 30,159; >20 sites *n* = 3,785. Training mix: 3,093,529 ≤20 (89.5%) versus 362,673 >20 (10.5%) of 3,456,202. The specialist comparison in main Fig. 5d–f is a different cohort (one frozen 500-structure >20 subset) and is not pooled with these bins.

**Supplementary Figure 7 | Full-test size-bin rates.**
Repository file: `fig_s4_size_bins.svg`.

**Supplementary Table 8 | Full-test Multi versus formula reference by size bin.**

| Lattice | Bin | *n* | Multi T1 | Formula T1 | Multi T20 | Formula T20 |
|---|---|---:|---:|---:|---:|---:|
| Free | ≤20 | 30,159 | 87.6 | 89.4 | 94.9 | 95.9 |
| Free | >20 | 3,785 | 55.4 | 58.1 | 64.4 | 69.3 |
| Known | ≤20 | 30,159 | 94.0 | 93.1 | 96.7 | 96.8 |
| Known | >20 | 3,785 | 61.9 | 61.4 | 69.2 | 70.2 |

**Supplementary Table 9 | Same 500-structure >20 subset as main Fig. 5d–f (reranked).**
Not pooled with Supplementary Table 8.

| Lattice | Method | T1 | T5 | T10 | T20 | T50 |
|---|---|---:|---:|---:|---:|---:|
| Free | Base Multi | 53.0 | 57.6 | 59.2 | 60.2 | 61.2 |
| Free | Specialist Multi | 66.4 | 72.0 | 73.6 | 74.4 | 75.6 |
| Free | Base formula | 58.4 | 64.2 | 64.8 | 65.8 | 67.6 |
| Known | Base Multi | 61.0 | 66.0 | 67.2 | 67.6 | 68.2 |
| Known | Specialist Multi | 72.6 | 75.6 | 76.8 | 78.4 | 79.2 |
| Known | Base formula | 61.0 | 64.8 | 66.2 | 67.2 | 68.4 |

---

## Supplementary Note 8. Physical rerank and unused ranking levers

The physical reranker scores only prefixes present in the query:

> *R* = 1.0 · *R*_PXRD + 0.25 · *R*_ND + 0.2 · *R*_charge + validity gate.

*R*_PXRD and *R*_ND are rasterized cosines on 5–80° (σ = 0.25°). Charge balance uses SMACT. Invalid cells (minimum distance < 0.5 Å or volume per atom outside 2–150 Å<sup>3</sup>) receive a penalty of −2. The neutron weight 0.25 was selected by a frozen-pool sweep: on noisy-exp Multi, lattice-free Top-1 is 73.0 / 74.2 / **74.6** / 73.8 / 73.6% at *w*_ND = 0, 0.10, 0.25, 0.50, 1.00 (Supplementary Fig. 8). The default 0.25 is the peak of that curve, not a post-hoc pick on CNRS/RRUFF.

For formula+lattice experimental head-to-heads we fuse physics with CHGNet energy:

> *F*_chgnet = rr(physics) + rr(−*E*_CHGNet), rr(*x*) = 1 / (10 + rank(*x*)).

Hyperparameters of this fusion (including a consensus cluster term) were compared on the independent temperature-selection split and did not beat the default; they are not retuned on CNRS or RRUFF.

**Tried and not used** (all checked on independent data, none promoted into the paper protocol):

- Space-group token branch decoding. The `<SG###>` token does not control site symmetry. Forcing the correct space-group token yielded 0/30 hits on the probe set.
- Relax-then-rerank with relaxed energy, post-relax spectrum consistency, or displacement, after removing label leakage: all ≤ the generated-state physics rank.
- Multi-temperature candidate pooling.
- Full-pattern *R*_wp, bond-valence sums, or CHGNet energy as a standalone ranker.
- An n = 13–20 specialist routed on CNRS: recall decreased relative to the base mix model.
- A 6-million-record corpus retrain.

**Supplementary Figure 8 | Neutron rerank weight on a frozen noisy Multi pool.**
Lattice-free Top-1, *N* = 500. Default *w*_ND = 0.25. Repository file: `fig_s14_rerank_nd_weight.svg`.

---

## Supplementary Note 9. Synthetic artefacts (companion to main Fig. 6)

“Noisy” in main Fig. 6 and here means online synthetic perturbations of simulated inputs: 2θ shift, peak broadening, intensity/texture, background and counting noise. It does not mean a measured pattern. Supplementary Fig. 9 restates the clean versus noisy high-information contrast. The system-level comparison is clean-trained + clean input versus noise-finetuned + noisy input.

**Supplementary Figure 9 | Clean versus noisy high-information Top-1.**
Repository file: `fig_s11_clean_vs_noisy.svg`. Numerical source: `clean_noisy_high_information.csv` (main Fig. 6).

| Lattice | Condition | Clean T1 | Noisy T1 | Clean T20 | Noisy T20 |
|---|---|---:|---:|---:|---:|
| Free | PXRD+XRF | 91.2 | 71.6 | 97.4 | 93.8 |
| Free | Multi | 91.4 | 75.4 | 97.2 | 92.4 |
| Free | Formula | 91.8 | 77.8 | 97.8 | 93.8 |
| Known | PXRD+XRF | 95.2 | 92.2 | 98.4 | 98.4 |
| Known | Multi | 96.4 | 92.2 | 98.6 | 97.8 |
| Known | Formula | 94.2 | 92.2 | 98.4 | 97.8 |

Deploy matching (σ = 0.40°, ±0.20° shift) is reported only for noisy, lattice-free pools. Rietveld-lite reorders a frozen pool; large-cell Top-20 does not move (main Fig. 6e,f). Ranking cannot recover a never-generated large cell. The faded XPS control in Fig. 6e (76.8% → 79.6% Top-1 on the frozen noisy Multi+XPS pool) remains below Multi (Supplementary Note 5).

---

## Supplementary Note 10. Wall-clock cost

Autoregressive sampling is used because it makes *N* = 100 candidates per query the default budget of every rate in the paper. Speed is not a title-level claim.

On an NVIDIA GeForce RTX 4090 D, 120 evaluation shards of a generation job at *N* = 100 have median 19.0 s per structure (10th percentile 13.1 s, 90th percentile 23.5 s) (Supplementary Fig. 10). These numbers describe generation plus parse and physical rerank on that hardware. They are not a comparison against a diffusion or flow solver on matched hardware, and they are not a contribution.

**Supplementary Figure 10 | Seconds per structure at *N* = 100.**
Median marked. Hardware: RTX 4090 D. Source: `runtime_summary.json`, `fig_s15_runtime.csv`. Repository file: `fig_s15_runtime_per_structure.svg`.

**Supplementary Table 10 | Runtime summary (*N* = 100).**

| Quantity | Value |
|---|---|
| Hardware | NVIDIA GeForce RTX 4090 D |
| Shards | 120 |
| Median | 19.0 s / structure |
| 10th percentile | 13.1 s |
| 90th percentile | 23.5 s |

---

## Supplementary Note 11. MP-20: two protocols, not one axis

MP-20 is not a main-text result. Literature generators report Top-1 on the official 9,046-structure test split. Our internal SFT check used the **first 500** MP-20 test records. Those are different populations. We do not plot our 83.2% (first 500) on the same axis as PXRDGen’s full 9,046. The repository drawing `fig_s10_mp20_protocol.svg` does exactly that and is **not** a supplementary figure of this paper. The two protocols are two tables.

A separate, valid observation is leakage: 6,399 of 9,046 official MP-20 test records (70.7%) are near-duplicates of a RealPXRD-style training pool; 2,647 (29.3%) are not (Supplementary Fig. 11). Any MP-20 number, including the literature rates below, should be read with that leakage in view. Our main ladder (Fig. 3) is the leakage-audited Tier-A split, not MP-20.

**Supplementary Figure 11 | MP-20 official test versus a RealPXRD-style training pool.**
Near-duplicate share 70.7% of *N* = 9,046. Repository file: `fig_s16_mp20_leakage.svg`.

**Supplementary Table 11 | Literature MP-20 Top-1 on the official full test (*N* = 9,046).**
Quoted from the published papers. Not a result of this work.

| Setting | Method | Top-1 (%) |
|---|---|---:|
| Lattice-free | PXRDGen | 68.68 |
| Lattice-free | Uni-3DAR | 75.08 |
| Known lattice | PXRDGen | 75.32 |
| Known lattice | XRDSol | 82.3 |

**Supplementary Table 12 | This work, MP-20 first 500 only.**
Internal SFT check. **Not comparable** to Supplementary Table 11.

| Setting | *n* | Top-1 (%) |
|---|---:|---:|
| Lattice-free | 500 | 83.2 |
| Known lattice | 500 | 87.2 |

---

## Supplementary Note 12. Temperature and top-*k* were locked off CNRS/RRUFF

Temperature 1.0 and top-*k* 50 for the experimental head-to-head were selected on an independent temperature-selection split (public and Tier-A slices). That split did not include CNRS-126, RRUFF-269, or the laboratory LiCoO2 pattern. The paper mix checkpoint, PXRD+formula, known lattice, *N* = 100, and `F_chgnet` were likewise frozen before the CNRS/RRUFF numbers were read. Refine-before-match (lattice-frozen CHGNet and MACE, fmax 0.05 eV Å<sup>−1</sup>, ≤300 steps; hit if generated or relaxed matches) was validated on the same independent split, where it did not decrease recall: public slice 22/34 generated → 25/38 after refine; Tier-A slice 162/178 → 163/179. Fusion hyperparameters, including a consensus cluster term, were compared on that split and did not beat the default.

---

## Supplementary Note 13. CNRS and RRUFF construction and the RPS dual

RealPXRD-Solver reports CNRS (*n* = 126) and RRUFF (*n* = 269) given formula and lattice, with GSAS-II on its candidates. Our head-to-head uses the same files and the dual protocol: autoregressive recall + lattice-frozen MLIP (main text; Supplementary Note 1). We always quote RPS paper counts and our reproduction of the released code.

**Supplementary Table 13 | CNRS and RRUFF versus RealPXRD-Solver.**
“Generated” is the language-model list before lattice-frozen MLIP. “This work” is after refine-before-match. Intervals and McNemar tests compare this work with reproduced RPS.

| Set | Cut-off | Generated | This work | 95% CI | Reproduced RPS | RPS paper | Δ vs reproduced (CI) | McNemar *p* |
|---|---|---:|---:|---|---:|---:|---|---:|
| CNRS (126) | Top-1 | 94 | 100 | 91–109 | 101 | 98 | −1 (−12, +10) | 1.0 |
| CNRS | Top-20 | 116 | 118 | 112–123 | 118 | 116 | 0 (−6, +6) | 1.0 |
| RRUFF (269) | Top-1 | 216 | 221 | 208–233 | 199 | 212 | +22 (+8, +36) | 0.003 |
| RRUFF | Top-20 | 230 | 246 | 237–255 | 245 | 250 | +1 (−9, +11) | 1.0 |

CNRS is statistically indistinguishable from reproduced RPS. RRUFF Top-1 is significantly ahead. RRUFF Top-20 is tied after refinement: before refinement, RRUFF Top-20 lagged the reproduced solver; lattice-frozen MLIP closes that coordinate gap. These four cells do not support a global statement that the language model outperforms RPS. An oracle that also counts never-generated structures that become matches only after relaxation is CNRS 121 → 122 and RRUFF 233 → 252; the paper rates use the generated-or-relaxed rule on the reported list, not that oracle.

RRUFF DIF coordinates are literature single-crystal sites; the powder pattern refines the cell. That is the published RRUFF convention. It is weaker than a same-sample Rietveld coordinate and applies to both solvers.

Construction gates used when assembling comparable experimental files (not accuracies) are in `relaxed_benchmark_scale.json` and are summarized below. The paper comparison uses the RPS-released 269-entry RRUFF list and 126-entry CNRS list, not the intermediate CNRS-98 / RRUFF-56 slices.

| Source | Parsed / DIF-built | Primitive *n* ≤ 20 | + inorganic | Cosine ≥ 0.40 | After dedup (earlier slice) | Paper list |
|---|---:|---:|---:|---:|---:|---:|
| CNRS (opXRD) | 548 | 207 | 205 | 172 | 98 | 126 |
| COD | 159 | 48 | 48 | 32 | 19 | — |
| HKUST-A | 20 | 16 | 16 | 11 | 11 | — |
| RRUFF | 314 | 87 | 87 | 85 | 56 | 269 |

RRUFF started from 1,484 patterns with DIF files; 790 DIF files had no atoms, 205 IDs had no DIF, and 69 failed a self-check. Those counts describe file construction.

---

## Supplementary Note 14. Laboratory powders other than LiCoO2

The main text reports one laboratory case, LiCoO2 (Fig. 7c). Nine further Bruker Cu Kα powders from the same collaborator set are recorded here as a case series, not as a generalization and not as a percentage. Neutron and XRF prefixes, where used, are CIF-simulated. “Noisy-exp” is the noise-adapted checkpoint; “clean” is the mix-family clean checkpoint.

A library audit against the 3,456,202 training structures (exact reduced formula, cap 500 same-formula entries, same `StructureMatcher`) found a structure match for 9 of the 10 targets. The exception is 2M-WS2, which is not in the library and is not recovered by the generator under any spectroscopic rung. PXRD-similarity retrieval on the same library is Top-1 for Li2CO3 and Li2SO4 (both unsolved by generation) and Top-50 for LiCoO2 (rank 8). Retrieval Top-1 on the ten is 2; Top-50 is 3. These numbers are a novelty audit, not a claim that the generator is a retrieval engine.

**Supplementary Figure 12 | Collaborator ranks, noise-adapted model, *N* = 10.**
Lattice-free (a) and known-lattice (b). Cells are best *k* or a miss. Repository file: `fig_s12_collaborator_ranks.svg`. Source: `per_sample_ranks.csv`.

**Supplementary Table 14 | Noise-adapted model, known lattice: best rank (100 candidates).**
A dash is a miss. ND is CIF-simulated. This table is a case series.

| Sample | Sites | In train library | PXRD | +ND | +XRF | Multi | Formula |
|---|---:|---|---:|---:|---:|---:|---:|
| LiCoO2 | 12 | yes | 2 | **1** | 1 | 1 | 1 |
| NaCoO2 | 12 | yes | — | — | 15 | **1** | 60 |
| BiOCl | 6 | yes | — | — | **1** | 1 | 1 |
| Nb3Cl8 | 22 | yes | — | — | — | — | **1** |
| 1T-TaS2 | 3 | yes | — | — | — | — | 2 |
| 2M-WS2 | 12 | **no** | — | — | — | — | — |
| Li2CO3 | 24 | yes | — | — | — | — | — |
| Li2SO4 | 28 | yes | — | — | — | — | — |
| Na3PS4 | 16 | yes | — | — | — | — | 3 |
| SiS2 | 12 | yes | — | — | — | 9 | — |

**Supplementary Table 15 | Noise-adapted model, lattice-free: best rank (100 candidates).**

| Sample | PXRD | +ND | +XRF | Multi | Formula |
|---|---:|---:|---:|---:|---:|
| LiCoO2 | — | 2 | **1** | 1 | 1 |
| NaCoO2 | — | — | 31 | 37 | 14 |
| BiOCl | — | — | 3 | 23 | 4 |
| Nb3Cl8 | — | — | — | — | — |
| 1T-TaS2 | — | — | — | — | — |
| 2M-WS2 | — | — | — | — | — |
| Li2CO3 | — | — | — | — | — |
| Li2SO4 | — | — | — | — | — |
| Na3PS4 | — | — | — | — | — |
| SiS2 | — | — | — | — | — |

**Supplementary Table 16 | Training-library novelty audit (lower bound).**
Exact reduced formula, cap 500 same-formula library entries, same `StructureMatcher`. Retrieval is PXRD cosine against that candidate list.

| Sample | Same-formula in library | Structure in library | Retrieval rank | Best cosine |
|---|---:|---|---:|---:|
| LiCoO2 | 21 | yes | 8 | 0.940 |
| NaCoO2 | 19 | yes | — | 0.815 |
| BiOCl | 9 | yes | — | 0.835 |
| Nb3Cl8 | 7 | yes | — | 0.915 |
| 1T-TaS2 | 44 | yes | — | 0.877 |
| 2M-WS2 | 25 | **no** | — | 0.985 |
| Li2CO3 | 9 | yes | **1** | 0.938 |
| Li2SO4 | 11 | yes | **1** | 0.985 |
| Na3PS4 | 9 | yes | — | 0.868 |
| SiS2 | 31 | yes | — | 0.883 |

Known-lattice Multi Top-1 occurs for LiCoO2, NaCoO2 and BiOCl. That count is not a 30% generalization. Li2CO3 and Li2SO4 are in the library and retrieval finds them; the generator misses both. 2M-WS2 is absent from the library (lower bound) and unsolved. The clean checkpoint misses LiCoO2 on every spectroscopic rung (main Fig. 7c); it can still hit BiOCl, Nb3Cl8 and 1T-TaS2 when mass fractions or a formula are given. Noise adaptation is what unlocks LiCoO2 from experimental PXRD plus simulated ND.

---

## Supplementary Note 15. Held-out experimental PXRD: protocol only

A broader experimental PXRD collection (COD / opXRD CNRS / HKUST-A) is being recoded onto the same primitive-cell and Cu Kα convention as the simulated corpus: infer axis and wavelength from the paired structure, remap to λ = 1.54184 Å, and apply the same profile front-end. Main Fig. 7b is reserved for that ladder. **No solver accuracy from this collection is reported in this draft.** Retired numbers from an earlier encoding are withdrawn.

What can be stated now is construction QC, not solver recall. Supplementary Fig. 13 is the distribution of simulated-PXRD cosine between the experimental pattern and the paired CIF: it measures whether the file pair is geometrically consistent, not whether the language model solved the structure. After the cosine ≥ 0.40 gate the assembled pairs (*n* = 337) have median cosine 0.90 (mean 0.84; 10th percentile 0.57; 90th percentile 0.98; sources: opXRD 281, COD 56). Supplementary Fig. 14 and Supplementary Table 17 count records removed **before** any model is run. Those filters define a hold-out; they are not a result.

**Supplementary Figure 13 | Pattern–CIF cosine for hold-out construction.**
QC of file pairing. Not a model accuracy. Repository file: `fig_s7_public_cosine_qc.svg`.

**Supplementary Figure 14 | Hold-out filter counts.**
Pre-model exclusions. Repository file: `fig_s8_public_filter_counts.svg`.

**Supplementary Table 17 | Pre-model exclusions used to construct the reserved hold-out.**
Source: `fig_s8_public_filter_counts.csv`. These are file-quality gates, not solver rates.

| Reason | *n* |
|---|---:|
| Too many sites | 491 |
| Cosine < 0.40 | 162 |
| No full structure | 161 |
| Occupancy collapsed | 91 |
| No observed PXRD | 74 |
| Too few PXRD bins | 44 |
| Other | 3 |

---

## Supplementary Note 16. Negative results and unused directions

These limits belong with the claims, not after them.

1. **No composition.** Lattice-free PXRD is 3.8% Top-1 on 33,944 simulated structures and a miss for clean-model LiCoO2. The inverse problem remains underdetermined.
2. **Large cells.** Specialist Multi still leaves about one in three >20-site queries unsolved at Top-1 (main Fig. 5). Rietveld-lite does not close that gap.
3. **Experimental neutron.** Every neutron prefix on an experimental PXRD in this paper is CIF-simulated. A domain gap to measured powder neutron diffraction is untested.
4. **XPS-style valence.** Near-zero increment once mass fractions are known (Supplementary Note 5).
5. **Unused levers.** Supplementary Note 8. We do not commit to a 6-million-record retrain, policy-gradient fine-tuning, or a coordinate-space flow head.
6. **Generation / design.** The same language can drop PXRD and sample under composition or lattice constraints. The experiments stop at inversion (main-text Discussion).

---

## Figure and table index

| Item | Maps to repository file (if any) | Main-text hook |
|---|---|---|
| Supplementary Fig. 1 | `fig_s13_v2_token_length.svg` | Methods, language |
| Supplementary Fig. 2 | `fig_s2_pxrd_vs_nd.svg` | Fig. 4a |
| Supplementary Fig. 3 | `fig_s6_governance_counts.svg` | Fig. 2 |
| Supplementary Fig. 4 | `fig_s1_simulated_ladder_top1.svg` | Fig. 3, XPS omitted |
| Supplementary Fig. 5 | `fig_s5_xps_increment.svg` | Fig. 3, Fig. 6e |
| Supplementary Fig. 6 | `fig_s3_nd_gain_by_system.svg` | Fig. 4c–f |
| Supplementary Fig. 7 | `fig_s4_size_bins.svg` | Fig. 5a,b |
| Supplementary Fig. 8 | `fig_s14_rerank_nd_weight.svg` | Methods, rerank |
| Supplementary Fig. 9 | `fig_s11_clean_vs_noisy.svg` | Fig. 6 |
| Supplementary Fig. 10 | `fig_s15_runtime_per_structure.svg` | Methods, speed |
| Supplementary Fig. 11 | `fig_s16_mp20_leakage.svg` | Fig. 2 SI pointer |
| Supplementary Fig. 12 | `fig_s12_collaborator_ranks.svg` | Fig. 7c |
| Supplementary Fig. 13 | `fig_s7_public_cosine_qc.svg` | Fig. 7b reserved |
| Supplementary Fig. 14 | `fig_s8_public_filter_counts.svg` | Fig. 7b reserved |
| Supplementary Tables 1–17 | see Notes | — |
