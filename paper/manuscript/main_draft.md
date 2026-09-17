<!--
DRAFT HEADER — remove before submission.

Wrote this round
  Title; Abstract; Introduction; Results 3.1–3.6 (Fig. 1–6); Results 3.7 CNRS/RRUFF
  (Fig. 7a) and LiCoO2 (Fig. 7c); Discussion (innovation 4 only in the last paragraph);
  Methods; Figure legends; Acknowledgements / Data / Code / Competing-interest placeholders.
  Supplementary Information: Notes 1–16, Figs. S1–S14, Tables 1–17 in
  paper/manuscript/supplementary.md. Main-text SI citations added in this pass.

Did not write
  Fig. 7b / any public-337 accuracy, ladder, or the retired 8.6% / unrefined 67/77
  numbers. No 337 refine job is promised or scheduled from this draft.

Figures
  paper/figures_final/ contained no refined Fig. 1–6 at writing (2026-09-13).
  Fig. 2–6 captions and in-text panel letters follow the repository panel maps in
  paper_figures/fig2_dataset_atlas through fig6_noise_robustness. Fig. 1 has no
  drawing package; the caption is the schematic implied by Results 3.1 and must be
  aligned if a later drawing splits panels. Manuscript Fig. 7 follows LOCKED.md
  (a CNRS/RRUFF; b reserved; c LiCoO2), not the repository 10-sample Fig. 7.

Number sources
  Funnel / split / test atlas — paper_figures/fig2_dataset_atlas/README.md,
    data/dataset_summary.json (unified 10,207,687 → Tier-A 3,524,185; train
    3,456,202 / valid 34,039 / test 33,944).
  Simulated ladder — paper_figures/fig3_multimodal_performance/data/figure_3_source_data.csv
    extracted from crystal_language/eval_results/ladder_tierA_s2_full_neutron/ladder_test_table.json
    (N=33,944, N_cand=100, topk_rerank). Rounded as in STATUS.md.
  Neutron paired effects — paper_figures/fig4_neutron_value/data/paired_neutron_effects.csv
    and crystal_system_neutron_effects.csv from
    crystal_language/eval_results/ladder_tierA_s2_10k/stats/ladder_stats.json
    (aligned N=10,000, 2,000-resample paired bootstrap; CIs are for that 10k subset).
  Large cells — paper_figures/fig5_large_cell/data/full_test_size_bins.csv and
    bigcell_same_subset.csv from
    crystal_language/eval_results/ladder_tierA_s2_full_neutron/size_bins_20.json and
    crystal_language/eval_results/bigcell_ab/oracle_same_subset_comparison.json.
  Synthetic artefacts — paper_figures/fig6_noise_robustness/data/clean_noisy_high_information.csv,
    deploy_rerank_free.csv, refinement_free.csv, bigcell_clean_noisy.csv.
  CNRS vs RPS — crystal_language/eval_results/chgnet/rps_cnrs126_bootstrap_refined.json.
  RRUFF vs RPS — crystal_language/eval_results/chgnet/rps_rruff269_bootstrap_refined.json.
  LiCoO2 ranks — paper_figures/fig7_experimental_pxrd/data/panels/fig7c_licoo2_ranks.csv.
  Runtime (Methods / SI only) — paper_figures/si_supplementary/data/runtime_summary.json.
  RPS protocol — 总体实施方案.md §8.12 (reported vs reproduced counts; refine-before-match).
-->

**Complementary X-ray and neutron probes for crystal-structure solution in one autoregressive language model**

TODO: Author list.

TODO: Affiliations.

TODO: Corresponding author.

# Abstract

Powder X-ray diffraction (PXRD) from polycrystalline samples is underdetermined: X-ray form factors poorly distinguish neighbouring atomic numbers and light atoms, including Li and H. Neutron powder diffraction and composition measurements supply complementary contrast, yet they are usually treated by separate solvers. We serialize crystals as a parseable language and treat PXRD, neutron diffraction and elemental mass fractions as droppable prefixes of one autoregressive model. On 33,944 simulated test structures, lattice-free PXRD recovers 3.8% at Top-1; PXRD with neutron diffraction and exact elemental mass fractions reaches 84.0%, approaching an exact-formula reference of 85.9%. With a known lattice the multimodal condition is 90.4% versus 89.5% for that reference. On experimental powders, the same protocol matches a dedicated open-source PXRD solver on CNRS-126 and RRUFF-269 after lattice-frozen interatomic-potential refinement: CNRS is statistically indistinguishable from the reproduced solver, RRUFF Top-1 is significantly ahead, and RRUFF Top-20 is tied. On laboratory Cu Kα data for LiCoO2, a noise-adapted model with known lattice moves the structure from rank 2 with PXRD alone to Top-1 after a CIF-simulated neutron pattern is added.

# Introduction

A powder PXRD pattern collapses a three-dimensional crystal onto a one-dimensional intensity trace. Peak overlap, preferred orientation and the monotonic rise of X-ray form factors with atomic number leave the inverse problem underdetermined: neighbouring elements, light atoms, and hydrogen or lithium sites are only weakly contrasted<sup>1,2</sup>. Neutron coherent scattering lengths are not monotonic in Z and can restore contrast at those sites<sup>3</sup>, while a composition measurement — even a list of elemental mass fractions — turns an open-ended generation problem into a constrained one. In a laboratory these probes arrive independently, and any one of them may be missing. A useful solver therefore has to accept PXRD, neutron powder diffraction and composition as optional inputs rather than as a fixed stack.

Generative models have begun to invert powder patterns directly. Diffusion and flow models conditioned on PXRD and composition, including PXRDGen, XRDSol and PXRDnet, produce candidate crystals that can be refined against the pattern<sup>4–6</sup>. RealPXRD-Solver (RPS) is a dedicated experimental-PXRD generator: it encodes *d*-spacing–intensity fingerprints, supports lattice-conditioned and lattice-free inference, and reports high recall on the CNRS and RRUFF experimental sets after GSAS-II coordinate refinement of its candidates<sup>7</sup>. Those systems treat PXRD as the spectroscopic channel. Separately, autoregressive crystal models such as CrystaLLM, deCIFer and Uni-3DAR serialize structures as token sequences and, in some cases, accept a PXRD condition<sup>8–10</sup>. None of them attach neutron powder diffraction as an independent physical channel inside the same generator, or train one set of weights to degrade when that channel — or the formula, or the lattice — is dropped.

Here we do that with a CIF-like crystal language whose continuous spectra sit outside the token stream. PXRD and neutron patterns are encoded as dense prefixes at placeholder tokens; elemental mass fractions enter as structured condition tokens. Formula-constrained decoding is used only when an exact formula is supplied. One decoder, trained on a mixture of missing-modality prefixes and of known versus unknown lattices, covers the combinations that a powder experiment actually produces (Fig. 1).

The scientific claims are three, and they are nested. First, a single autoregressive language can condition on droppable X-ray, neutron and composition prefixes without a separate architecture per probe. Second, on a leakage-audited simulated test of 33,944 structures, those prefixes form an information ladder: composition is what makes the inverse problem solvable, and neutron diffraction supplies a measurable gain where PXRD is weakest, including a paired increment that remains significant given composition once the lattice is known. Third, the same model, under a protocol aligned with a dedicated PXRD solver (formula, known lattice, refine-before-match), matches that solver’s recall on CNRS-126 and RRUFF-269, and a noise-adapted copy of the same language recovers laboratory LiCoO2 when a CIF-simulated neutron pattern is added to a measured Cu Kα trace. Speed is not a title-level claim; autoregressive sampling is used because it makes *N* = 100 candidates per query routine, which is the operating point of every rate reported below.

We do not claim to have introduced crystal language models, and we do not claim to be more accurate than a flow solver as a general statement. The question is whether one representation can walk the probe ladder and still stand on the experimental PXRD benchmarks that a specialist solver already occupies.

# Results

## Missing-modality crystal language

A crystal is written as a deterministic, CIF-like token sequence: formula, site count, space group, lattice, then one token triple per site (Fig. 1). In the v2 representation used throughout this paper, fractional coordinates are 1/1000 bins (`<C000>`–`<C999>`), lattice lengths are 0.05 Å bins and lattice angles are 0.5° bins; space groups occupy `<SG001>`–`<SG230>`. The grammar is parsed by a single decoder, so every sample either yields a valid structure or is discarded.

PXRD and neutron powder patterns never enter that discrete stream. Each pattern is converted to a variable-length list of (*d*, intensity) peaks and read by a small permutation-equivariant encoder into a fixed number of prefix embeddings, which replace `<PXRD_EMB>` and `<ND_EMB>` placeholders. Encoding *d* rather than a 2θ raster removes the diffractometer geometry from the prefix: a pattern collected at another wavelength is the same object after conversion to *d*. Elemental composition, when given, is not a measured X-ray fluorescence (XRF) spectrum. It is a list of exact elemental mass fractions serialized as `<XRF_MASS_FRAC>` tokens. An exact-formula condition is a stronger, separate prefix: the formula block is copied into the prompt and a grammar mask then forces element identities and counts, leaving the model free only on the numeric tokens it must actually predict.

Any of those prefixes can be absent. Training draws the condition set from {PXRD, PXRD+elements, PXRD+XRF, PXRD+ND, PXRD+ND+XRF, PXRD+formula} and independently omits the lattice with probability 1/2, so one set of weights covers the missing-modality combinations and the known-lattice / lattice-free split. The paper model is a 16-layer decoder (hidden size 1024, 16 heads; 386 million parameters) with RMSNorm, rotary positional embeddings and SwiGLU. Because decoding is autoregressive, drawing *N* = 100 candidates per query is the default evaluation budget used for every Top-*k* rate below; wall-clock figures are in Supplementary Note 10 and Supplementary Fig. 10, not as a contribution. Grammar length on the test set is in Supplementary Note 2 and Supplementary Fig. 1.

After sampling, candidates are re-simulated and ranked by physical consistency with the prefixes the model actually received (PXRD cosine, neutron cosine if a neutron prefix was present, charge balance, and a validity gate). When an exact formula is available we further fuse that physics rank with a CHGNet energy rank. Experimental head-to-heads then apply a lattice-frozen interatomic-potential relaxation to the reported list and count a hit if either the generated or the relaxed structure matches. Discrete tokens carry topology and composition; the potential supplies metric coordinates. That pairing is the dual of a flow generator followed by GSAS-II, not a claim that the language model is a more accurate crystallographer.

## A leakage-audited multimodal corpus

The prefixes are only as trustworthy as the records behind them (Fig. 2; Supplementary Note 4 and Supplementary Fig. 3). We start from 10,207,687 unified multimodal records assembled from Alexandria, AFLOW, OQMD, WBM and the Materials Project, then keep 4,127,590 Tier-A quality records and 3,524,185 after tight structure-level deduplication (Fig. 2a). The frozen split is train 3,456,202 / valid 34,039 / test 33,944. After deduplication the kept mix is Alexandria 62.6%, AFLOW 15.3%, OQMD 10.8%, WBM 7.1% and Materials Project 4.2% (Fig. 2b). The scientific product is not the union of those source names. It is a governed, leakage-audited split in which every kept record carries a PXRD pattern, a paired neutron pattern and elemental mass fractions simulated under one physical convention.

All simulated patterns use Cu Kα (λ = 1.54184 Å) over 5–80° 2θ. Neutron patterns are generated with tabulated coherent scattering lengths on the same angular window so that X-ray and neutron prefixes are paired, not independently rescaled. XRF tokens store composition-derived mass fractions. Train, validation and test assignments were frozen after a two-stage leakage audit (coarse composition–lattice buckets, then `StructureMatcher` confirmation); the test scan of 33,944 records is complete, not a subsample (Fig. 2c–e). Test cells have a median of 10 sites (mean 12.8, 95th percentile 36, range 1–128) and a median of three distinct elements. Crystal systems span the standard space-group ranges; monoclinic, orthorhombic and tetragonal together account for about two thirds of the test set. External MP-20 comparisons, including train–test leakage of 70.7% of the official 9,046-structure test against a RealPXRD-style training pool, are reported only in Supplementary Note 11 and Supplementary Fig. 11 and are not plotted against the full-test ladder in the main figures.

## Information ladder on simulated powders

The inverse problem is solvable once composition is present, and PXRD+ND+XRF approaches an exact-formula reference; it does not require that reference as an input (Fig. 3). All rates in this section are reranked Top-*k* on the full Tier-A test set (*N* = 33,944 structures, *N*_cand = 100; protocol in Supplementary Note 1). The five conditions are PXRD, PXRD+ND, PXRD+XRF, PXRD+ND+XRF (Multi), and PXRD plus the exact formula (formula reference). The formula condition is an oracle reference: it tells the model the correct stoichiometry. It is not a theoretical upper bound, because a multimodal prefix can in principle outrank a formula-only sample on physical consistency. Full Top-*k* and composition-recovery tables are Supplementary Table 6. XPS-style valence tokens are omitted here because the increment is near zero (Supplementary Note 5, Supplementary Figs. 4, 5).

Without a lattice, PXRD alone recovers 3.8% of structures at Top-1 (15.2% at Top-20). Adding a paired neutron pattern lifts Top-1 to 7.8% (Top-20 21.5%). Exact elemental mass fractions change the problem: PXRD+XRF reaches 83.2% Top-1, and Multi reaches 84.0%, against a formula reference of 85.9% (Fig. 3b). At Top-20 the same lattice-free comparison is Multi 91.5% versus formula 93.0%. Composition recovery tracks the structure-match jump: lattice-free PXRD recovers the composition in 24.8% of queries, PXRD+ND in 31.6%, and both XRF and Multi in 98.9% (Fig. 3d). Neutron diffraction without composition is a real but limited increment; composition is what turns generation into retrieval of a constrained crystal.

With a known lattice the same ladder sits higher (Fig. 3c). PXRD Top-1 is 7.1% (Top-20 24.7%) and PXRD+ND is 15.5% (Top-20 33.2%). Multi reaches 90.4% Top-1 and 93.6% Top-20; the formula reference is 89.5% and 93.9%. Multi therefore sits 0.8 percentage points above the formula reference at known-lattice Top-1 and 0.2 percentage points below it at Top-20 (Fig. 3f). Those gaps are the reason the formula condition is reported as a reference rather than as a bound: a spectroscopic prefix can outrank an exact formula on a subset of queries, particularly when the lattice is given and neutron contrast is available. Across Top-*k*, Multi and the formula reference converge once *k* is large (Fig. 3e). The practical operating point is Top-1 after rerank, which is the quantity carried into the neutron, large-cell and experimental sections.

## Where neutron diffraction contributes

Neutron diffraction is not a second PXRD channel. X-ray intensities weight electron density; neutron intensities weight nucleus-dependent coherent scattering lengths, which are non-monotonic in Z and remain finite for H and Li (Fig. 4a). The traces in Fig. 4a are schematic. Quantitative claims use an aligned 10,000-structure subset of the same test distribution, with paired bootstrap intervals (2,000 resamples) that describe that subset, not the full 33,944.

Adding ND to PXRD raises Top-1 by 4.61 percentage points without a lattice (95% CI 4.00–5.24) and by 8.93 points with a known lattice (8.18–9.71) (Fig. 4b). Both intervals exclude zero. The increment given XRF is smaller, as the full-test ladder already suggested. Lattice-free, ND given XRF is +0.37 points with a 95% CI of −0.07 to +0.82, which crosses zero. With a known lattice the same paired contrast is +0.68 points (0.40–0.97) and is significantly positive. Neutron diffraction therefore does independent work on top of PXRD, and still does measurable work on top of XRF once the cell is known. It does not, on this subset, deliver a significant lattice-free increment once exact mass fractions are already in the prefix. That limitation is part of the result, not a caveat attached afterwards.

Crystal-system breakdowns are descriptive point estimates: the source artefact does not supply subgroup confidence intervals (Fig. 4c–f; every system in Supplementary Table 7 and Supplementary Fig. 6). The largest PXRD→+ND gains are in hexagonal cells (+10.0 points lattice-free, +15.4 known), then cubic and, depending on the lattice setting, trigonal or tetragonal. Triclinic cells remain difficult even with ND. ND-given-XRF cells are near zero and change sign across systems, consistent with the aggregate lattice-free interval that crosses zero. The collection-level statement is the paired interval in Fig. 4b, not a ranking of crystal systems.

## Large-cell recall

The ladder averages over a training distribution that is dominated by small cells. Of 3,456,202 training crystals, 3,093,529 (89.5%) have ≤20 sites and 362,673 (10.5%) have more (Fig. 5c). On the full test set the drop is correspondingly sharp (Fig. 5a,b; Supplementary Note 7, Supplementary Fig. 7, Supplementary Table 8). Lattice-free Multi Top-1 is 87.6% for ≤20 sites (*n* = 30,159) and 55.4% for >20 sites (*n* = 3,785); the formula reference is 89.4% and 58.1%. Known-lattice Multi is 94.0% versus 61.9% in the same bins. Top-20 follows the same split. Large cells are a recall problem: the correct topology is often never sampled.

Routing at inference by site count — ≤20 to the base mix model, >20 to a specialist trained on large cells, same architecture — is a capacity allocation, not a new generator. All specialist comparisons below use one frozen 500-structure >20 subset and are not pooled with the full-test bins (Supplementary Table 9). On that subset, lattice-free specialist Multi Top-1 is 66.4% against 53.0% for the base Multi model and 58.4% for the base formula reference (Fig. 5d). Known-lattice specialist Multi Top-1 is 72.6% against 61.0% for both the base Multi model and the base formula reference (Fig. 5e). The specialist therefore exceeds the formula reference on the same large cells, by 8.0 points without a lattice and 11.6 points with one. Gains persist across Top-*k* (Fig. 5f). The remaining miss rate is still a generation coverage gap: ranking and relaxation cannot recover a structure that was never proposed.

## Robustness to synthetic artefacts

Laboratory powders are not the simulated traces used in Fig. 3–5. Before any experimental claim we therefore ask whether the language still ranks crystals when the prefixes are corrupted on purpose. In this section, and in Fig. 6, “noisy” means online synthetic artefacts applied to simulated patterns — 2θ shift, peak broadening, intensity/texture changes, background and counting noise. It does not mean a measured diffractogram.

Every configuration uses *N* = 500. The clean/noisy contrast is a system-level comparison: a clean-trained model on clean input versus a noise-finetuned model on noisy input. It is not a pure input-noise ablation. Under lattice-free inference, noisy Top-1 on the high-information rungs falls to 71.6% (PXRD+XRF), 75.4% (Multi) and 77.8% (formula reference), from clean values of 91.2%, 91.4% and 91.8% (Fig. 6b). Known-lattice Top-1 remains above 92%. Top-20 on the displayed noisy conditions stays at or above 92.4% (Fig. 6c). Recall of the correct topology is therefore more robust than Top-1 ranking. A clean model presented with noisy prefixes is not a usable composition-conditioned solver; noise-adapted weights are a deployment prerequisite, not an optional extra.

Two ranking adjustments were tested on frozen candidate pools so that generation coverage is held fixed. A deploy matching geometry (raster σ = 0.40° with a ±0.20° shift search, versus baseline σ = 0.25°) is reported only for noisy, lattice-free inference and lifts Top-1 by 2.8 points on PXRD+XRF and Multi and by 0.4 points on the formula reference (Fig. 6d). A lightweight profile-residual reranker (Rietveld-lite) raises base-model Top-1 by a mean of 3.3 points on the same frozen pools and by 1.4 points on the large-cell specialist pool; Top-20 on the large-cell pool does not move (Fig. 6e,f). Ranking can promote a near miss. It does not invent a large cell that the decoder never emitted. XPS-style valence tokens are omitted from the main ladder; a faded control in Fig. 6e is recorded in Supplementary Note 5, where the increment is near zero. Companion clean versus noisy rates are Supplementary Fig. 9 and Supplementary Note 9.

## Experimental powders

The simulated ladder uses patterns generated from the same CIF that defines the target. Experimental PXRD does not. We therefore evaluate the paper mix model on two public experimental sets under the protocol a dedicated PXRD solver already occupies, and then ask whether a noise-adapted copy of the same language can use a neutron prefix on a laboratory Cu Kα pattern. A broader experimental hold-out is reserved for a later report (Fig. 7b; construction protocol only in Supplementary Note 15).

**CNRS and RRUFF versus RealPXRD-Solver.** RPS reports Top-1 / Top-20 on CNRS (*n* = 126) and RRUFF (*n* = 269) given formula and lattice, with GSAS-II refinement of its candidates<sup>7</sup>. We align to that setting: paper mix model, PXRD+formula, known lattice, temperature 1.0 and top-*k* 50 (both locked on an independent temperature-selection split that never touched CNRS or RRUFF; Supplementary Note 12), *N* = 100, physics+CHGNet fusion ranking (`F_chgnet`), and refine-before-match on the reported Top-20. Lattice-frozen relaxations with CHGNet and MACE-MPA-0 (fmax 0.05 eV Å<sup>−1</sup>, ≤300 steps) are counted as hits if either the generated structure or a relaxed structure matches. Autoregressive recall plus a frozen-cell potential is the dual of RPS generation plus GSAS-II, not a claim of a better crystallographic residual. We always quote both the RPS paper counts and our reproduction of the released RPS code on the same files. Differences of one or two structures are read from paired bootstrap intervals, not from the sign of a single count.

On CNRS, refine-before-match Top-1 is 100 of 126 (bootstrap 95% CI 91–109) against a reproduced RPS of 101 (paper 98). The paired difference is −1 (CI −12 to +10; McNemar *p* = 1.0). Top-20 is 118 versus reproduced 118 (paper 116); difference 0 (CI −6 to +6). CNRS is statistically indistinguishable from reproduced RPS at both cut-offs (Fig. 7a). On RRUFF, Top-1 is 221 of 269 (CI 208–233) against reproduced 199 (paper 212). The paired difference is +22 (CI +8 to +36; McNemar *p* = 0.003). Top-20 is 246 versus reproduced 245 (paper 250); difference +1 (CI −9 to +11; *p* = 1.0). RRUFF Top-1 is significantly ahead of the reproduced solver. RRUFF Top-20 is tied after refinement. These four cells do not support a global statement that the language model outperforms RPS (Supplementary Note 13 and Supplementary Table 13).

**Laboratory LiCoO2.** A Bruker laboratory Cu Kα powder pattern of LiCoO2 was Q-aligned to λ = 1.54184 Å and rasterized onto 5–80°. The neutron prefix is simulated from the paired CIF with the same neutron calculator used in training. It is not an experimental neutron measurement. The clean mix model misses LiCoO2 on every spectroscopic rung, with or without a lattice (Fig. 7c). The noise-adapted model with known lattice ranks the correct structure 2nd from PXRD alone and 1st from PXRD+ND, without requiring XRF mass fractions. Lattice-free, the same noise-adapted model still misses on PXRD and first places LiCoO2 in the Top-10 only after the simulated neutron prefix is added. Noise-adapted weights are again the difference between a miss and a retrieval; the neutron prefix is the difference between rank 2 and Top-1 when the lattice is known. Other laboratory powders from the same collaborator set are recorded as a case series in Supplementary Note 14 (Supplementary Fig. 12, Supplementary Tables 14–16) and are not used as a main-text generalization.

# Discussion

Powder PXRD is underdetermined, and the missing information is not all of one kind. Composition turns an open generator into a constrained decoder. Neutron diffraction supplies a different physical contrast, which is visible as a paired Top-1 gain over PXRD and, when the lattice is known, as a smaller but significant gain on top of exact mass fractions. One autoregressive language can carry those prefixes because the spectra are not tokens: they are droppable dense conditions on a grammar that already knows how to emit a crystal. The simulated ladder is the mechanism. PXRD+ND+XRF approaches an exact-formula reference on 33,944 structures (lattice-free Top-1 84.0% versus 85.9%; known lattice 90.4% versus 89.5%). The formula condition remains a reference, not a bound.

On experimental powders the same weights, given formula and lattice, match a dedicated PXRD solver on that solver’s own CNRS and RRUFF splits after a lattice-frozen potential is allowed to finish the coordinates. Discrete tokens get the topology into the list; the potential, like GSAS-II in the RPS pipeline, supplies metric precision. CNRS is tied with the reproduced solver. RRUFF is ahead at Top-1 and tied at Top-20. That is a match on a specialist’s home ground, not a general accuracy ranking.

The failures are as informative as the hits. Without composition, lattice-free PXRD remains a 3.8% Top-1 problem on simulated data and a miss on laboratory LiCoO2 for the clean model. Large cells are a coverage gap: even the specialist leaves roughly one in three >20-site queries unsolved at Top-1, and Rietveld-lite does not close that gap. Neutron prefixes in every experimental panel of this paper are CIF-simulated; a domain gap to measured powder neutron diffraction is untested here. An XPS-style valence probe adds nothing once mass fractions are known (Supplementary Note 5). Unused ranking levers are listed in Supplementary Note 8. These are scope limits, not deferred extras (Supplementary Note 16).

The same language and constrained decoder are also a low-cost vehicle for conditional generation and design: any subset of prefixes can be dropped, including dropping PXRD entirely and sampling crystals under composition or lattice constraints. The experiments in this paper stop at inversion. We do not report de novo materials discovery, and we do not commit to a larger-corpus retrain, policy-gradient fine-tuning, or a coordinate-space flow head as a requirement for the claims above.

# Methods

**Crystal language.** Targets are serialized in a CIF-like grammar: `<CRYSTAL_START> <FORMULA> … <N_SITES> … <SPACE_GROUP> … <LATTICE> … <ATOM_SITES_START> … <CRYSTAL_END>`. The v2 vocabulary uses 1,000 fractional-coordinate bins, length bins of 0.05 Å from 1–50 Å, angle bins of 0.5° from 30–150°, and 230 space-group tokens. Sites are sorted by element then coordinates so that the sequence is deterministic. The tokenizer used with the paper checkpoint has 2,580 tokens. A generated sequence is parsed to a `pymatgen` `Structure` or rejected<sup>11</sup>. Target-token length versus site count on the test set is Supplementary Fig. 1 (Supplementary Note 2).

**Spectra prefixes.** PXRD and neutron patterns are stored as peak coordinates and intensities, converted to (*d*, *I*) lists (maximum 128 peaks) and encoded by a three-layer peak-set transformer into 16 prefix slots, injected at `<PXRD_EMB>` and `<ND_EMB>`. Missing modalities zero those slots. XRF is a token block of exact elemental mass fractions to three decimals, not an energy-dispersive spectrum. Simulated PXRD uses `pymatgen` `XRDCalculator`; simulated neutron patterns use `NDCalculator` at λ = 1.54184 Å over 5–80° so that the two prefixes share an angular window<sup>11,12</sup> (Supplementary Note 3). Experimental PXRD used in Fig. 7 is remapped onto that same Cu Kα window; any neutron prefix in Fig. 7 is regenerated from the paired CIF, never from a neutron diffractometer.

**Model and training.** The paper checkpoint is a decoder-only Transformer (16 layers, 16 heads, hidden size 1024, block size 1,024, dropout 0.1, rotary base 10,000, coordinate-value Fourier embeddings, conditional RMSNorm; 385,764,352 parameters) trained on the Tier-A v2 mix with the d–I encoder. Condition sets are sampled from PXRD, PXRD+element set, PXRD+XRF, PXRD+ND, PXRD+ND+XRF, PXRD+ND+XRF+XPS and PXRD+formula; the lattice is omitted with probability 0.5. XPS prefixes are present in the mix for completeness and are not used in the main-text ladder (Supplementary Note 5). The mix checkpoint (`crystal_language/runs/tierA_s2_di`) is the source of the simulated ladder, the neutron 10k subset, and the CNRS/RRUFF head-to-head; it was not modified for this manuscript. A large-cell specialist of the same architecture was trained on cells with >20 sites and is used only when the query has >20 sites. Noise-adapted weights used in Fig. 6 and for LiCoO2 were finetuned from the mix model with the synthetic artefact profile described below. Training hyperparameters and wall-clock measurements (median 19.0 s per structure at *N* = 100 on an NVIDIA GeForce RTX 4090 D) are in Supplementary Note 10 and Supplementary Table 10.

**Decoding.** Unless noted, we sample *N* = 100 candidates per query. CNRS/RRUFF and the experimental LiCoO2 case use temperature 1.0 and top-*k* 50, selected on an independent temperature-selection split (public and Tier-A slices) that did not include CNRS, RRUFF or the laboratory sample (Supplementary Note 12). Formula-constrained decoding is applied only when the prompt contains an exact formula: element tokens and counts are forced, and the model samples only the free numeric tokens. Known-lattice prompts inject the six lattice parameters as condition tokens; lattice-free prompts omit them and the decoder emits lattice bins.

**Physical rerank.** Candidates are re-simulated on 5–80° and scored only against prefixes present in the query: rasterized PXRD cosine (σ = 0.25°), neutron cosine with weight 0.25 when a neutron prefix is present, a SMACT charge-balance term (weight 0.2), and a validity gate (minimum interatomic distance 0.5 Å; volume per atom 2–150 Å<sup>3</sup>). Invalid candidates receive a penalty of −2. Near-duplicates are collapsed before ranking. The neutron weight 0.25 is the peak of a frozen-pool sweep (Supplementary Note 8, Supplementary Fig. 8). Deploy matching (Fig. 6d) uses σ = 0.40° and a ±0.20° shift search on noisy, lattice-free pools only. Rietveld-lite (Fig. 6e) is a lattice-frozen, differentiable profile-residual coordinate step applied to a frozen candidate pool; it reorders, it does not regenerate.

**Fusion rank and refine-before-match.** For formula+lattice experimental head-to-heads, candidates are ranked by `F_chgnet` = rr(physics) + rr(−*E*_CHGNet) with rr(*x*) = 1 / (10 + rank(*x*))<sup>13</sup>. The reported Top-20 are then relaxed with the cell frozen using CHGNet and MACE-MPA-0 (fmax 0.05 eV Å<sup>−1</sup>, ≤300 steps)<sup>13,14</sup>. A query is a hit if the generated structure or any relaxed structure matches the reference. This is the dual of RPS generating candidates and refining them with GSAS-II<sup>7,15</sup>.

**Structure matching.** All structure-match rates use `pymatgen` `StructureMatcher` with `ltol=0.3`, `stol=0.5`, `angle_tol=10` and `primitive_cell=True`<sup>11</sup> (Supplementary Note 1). Composition recovery is an exact reduced-formula match after parse.

**RPS reproduction.** RPS paper counts on CNRS and RRUFF are quoted alongside a reproduction obtained by running the authors’ released code on the same experimental files (`realpxrd_solver_si/*_allL`). CNRS paper: Top-1 98, Top-20 116. CNRS reproduced: 101 and 118. RRUFF paper: 212 and 250. RRUFF reproduced: 199 and 245. Paired bootstrap intervals and McNemar *p* values are from `rps_cnrs126_bootstrap_refined.json` and `rps_rruff269_bootstrap_refined.json` (2,000 resamples). Generated-versus-refined counts are Supplementary Table 13. RRUFF atomic coordinates in the DIF files are literature single-crystal sites; the powder pattern refines the cell. That is the published RRUFF convention and is weaker than a same-sample Rietveld coordinate, which we treat as a limitation of the benchmark rather than of one solver.

**Laboratory LiCoO2.** The pattern is a Bruker RAW1.01 Cu Kα scan, Q-aligned to λ = 1.54184 Å and rasterized onto 5–80°. XRF mass fractions, when used, are occupancy-weighted CIF compositions. Neutron intensities are `NDCalculator` simulations from the paired CIF. Clean-model ranks use the mix-family clean checkpoint; noise-adapted ranks use `tierA_s2_noisy_exp`. Neither neutron nor XRF in this case is a measured spectrum.

**Held-out experimental PXRD.** A public experimental PXRD collection recoded onto the same primitive-cell and Cu Kα convention will be reported separately. No accuracy from that collection is used in this draft (Supplementary Note 15).

**Statistics.** Full-test ladder rates are exact fractions of 33,944. Neutron intervals are paired bootstrap 95% CIs on the aligned 10,000-structure subset. Crystal-system neutron gains are point estimates. CNRS/RRUFF intervals are paired bootstrap 95% CIs on the per-structure hit vectors versus reproduced RPS; McNemar tests are two-sided. We never interpret a one-structure difference as a ranking.

# Supplementary information

Supplementary Notes 1–16, Supplementary Figs. 1–14 and Supplementary Tables 1–17 accompany this paper (`paper/manuscript/supplementary.md`).

# Data availability

TODO: governed Tier-A sequences, frozen split identifiers, simulated PXRD/ND/XRF records, CNRS/RRUFF file lists, LiCoO2 pattern and paired CIF, and source-data tables for Figs. 2–6 will be deposited on publication.

# Code availability

TODO: crystal-language training and evaluation code, rerank and refine-before-match scripts, and figure-generation notebooks will be released on publication. RealPXRD-Solver is used under its public licence<sup>7</sup>.

# Acknowledgements

TODO: funding, compute, experimental PXRD collection for LiCoO2, and people.

# Author contributions

TODO.

# Competing interests

TODO.

# Figure legends

**Figure 1 | Missing-modality crystal language.** Crystals are serialized as a CIF-like token sequence (formula, space group, lattice, sites). PXRD and neutron powder patterns are encoded as dense (*d*, *I*) prefixes at placeholder tokens and do not enter the discrete token stream. XRF enters as elemental mass fractions, not as a measured energy spectrum. Any prefix can be dropped; formula-constrained decoding is used only when an exact formula is supplied. One set of weights covers missing-modality combinations and known- or unknown-lattice inference. *Schematic; panel letters to be aligned with the user drawing if it splits this overview.* Token-length statistics: Supplementary Fig. 1.

**Figure 2 | Governance and coverage of the multimodal corpus.** **a**, Funnel from 10,207,687 unified records to 4,127,590 Tier-A records and 3,524,185 after tight deduplication, then the frozen split (train 3,456,202, valid 34,039, test 33,944). **b**, Source mix of the 3,524,185 kept records. **c**, Element prevalence on the complete test set (*N* = 33,944; one count per element per record). **d**, Site-count histogram (median 10; 95th percentile 36). **e**, Crystal-system distribution from space-group tokens. **f**, Schematic of a multimodal record: paired PXRD/ND intensity traces, XRF as structured mass fractions (not a raw energy spectrum), and crystal target tokens. Sources: `metadata_index_summary.json` (a, b); `export_summary.json` (split sizes); complete `test.jsonl` scan (c–e). Count tables: Supplementary Note 4, Supplementary Fig. 3, Supplementary Tables 2–4. MP-20 leakage is Supplementary Fig. 11, not this figure.

**Figure 3 | Information ladder on the full simulated test set.** All rates are reranked Top-*k* on *N* = 33,944 structures with *N*_cand = 100. Formula reference is an oracle that supplies the exact formula; it is not a theoretical upper bound. **a**, Ladder schematic: PXRD branches through ND (structural contrast) and XRF (elemental constraint) into Multi (PXRD+ND+XRF), compared with the formula reference. **b**, Lattice-free Top-1 and Top-20. **c**, Known-lattice Top-1 and Top-20. **d**, Composition recovery versus reranked Top-1; marker shape encodes lattice setting. **e**, Top-*k* curves for Multi and the formula reference. **f**, Multi minus formula-reference gaps at Top-1 and Top-20, and the Top-1 gain of Multi over PXRD. Source: `ladder_test_table.json` via `figure_3_source_data.csv`. XPS is omitted (Supplementary Note 5, Supplementary Figs. 4, 5). Full Top-*k*: Supplementary Table 6.

**Figure 4 | Value of neutron diffraction.** **a**, Schematic of complementary X-ray (electron-density, Z-weighted) and neutron (nuclear, isotope-sensitive) contrast. Traces are illustrative and are labelled as not data (Supplementary Fig. 2). **b**, Paired ΔTop-1 for ND versus PXRD and ND given XRF, lattice-free and known-lattice. Error bars are 95% paired bootstrap CIs on the aligned *N* = 10,000 subset (2,000 resamples), not full-test intervals. Lattice-free ND given XRF crosses zero; the other three effects are positive. **c**, Crystal-system heatmap of Top-1(PXRD+ND) − Top-1(PXRD), in percentage points. **d**, Crystal-system heatmap of Top-1(PXRD+ND+XRF) − Top-1(PXRD+XRF). **e**, PXRD baseline Top-1 versus ND-versus-PXRD gain; bubble size is the number of structures in the system. **f**, Largest crystal-system gains (descriptive point estimates; no subgroup CIs in the source) and the aggregate ND-given-XRF inference. Gains are percentage points. Source: `ladder_stats.json`. Every system: Supplementary Table 7, Supplementary Fig. 6.

**Figure 5 | Large-cell specialization.** **a**, Full-test reranked Top-1 (*N* = 33,944) for ≤20 (*n* = 30,159) and >20 (*n* = 3,785) cells, base Multi versus base formula reference, lattice-free and known-lattice. **b**, The same bins at Top-20. **c**, Training imbalance (3,093,529 ≤20 versus 362,673 >20 of 3,456,202) and the inference routing rule. **d**, One frozen >20 subset (*n* = 500), lattice-free: base Multi, specialist Multi, base formula reference. **e**, The same 500 with known lattice. Specialist Multi Top-1 is 66.4% versus formula 58.4% (free) and 72.6% versus 61.0% (known). **f**, Specialist gains over base Multi and over the base formula reference at Top-1/5/10/20/50 on that same subset. Panels a,b and d–f are different cohorts and are not pooled. Sources: `size_bins_20.json` (a, b); `oracle_same_subset_comparison.json` (d–f). Companion: Supplementary Fig. 7, Supplementary Tables 8, 9.

**Figure 6 | Robustness to synthetic experimental artefacts.** “Noisy” means online synthetic perturbations of simulated inputs, not measured patterns. All results *N* = 500 per configuration. Clean/noisy ladders compare a clean-trained model on clean input with a noise-finetuned model on noisy input. **a**, Qualitative schematic of the artefact pipeline (shift, broadening, intensity/texture, background, counting noise); traces are not data. **b**, Clean versus noisy Top-1 for PXRD+XRF, Multi and the formula reference. **c**, Corresponding Top-20; noisy recall stays ≥92.4% on the displayed rungs. **d**, Baseline matching (σ = 0.25°) versus deploy matching (σ = 0.40°, ±0.20° shift search) on the same frozen candidates; noisy and lattice-free only. **e**, Rietveld-lite reranking within each frozen refinement pool (top 25). A faded XPS control is shown for completeness and is not a main-text increment (Supplementary Note 5). The large-cell row is a separate pool. **f**, Clean versus noisy large-cell specialist rates and the three robustness takeaways. Deploy and refinement deltas use only their own frozen pools. Sources: `noisy_robustness_summary.json`, `noisy_rerank_deploy_base_ladder.json`, `refine_rerank_ablation_base_free.json`, `refine_rerank_ablation_bigcell_free.json`. Companion: Supplementary Fig. 9, Supplementary Note 9.

**Figure 7 | Experimental powders.** **a**, CNRS (*n* = 126) and RRUFF (*n* = 269) versus RealPXRD-Solver under PXRD+formula, known lattice, *N* = 100, `F_chgnet`, refine-before-match. Bars (to be drawn) show this work after refinement, reproduced RPS, and RPS paper counts, with paired-bootstrap intervals on this work. CNRS Top-1 100 [91, 109] versus reproduced 101 (paper 98); Top-20 118 versus 118 (paper 116). RRUFF Top-1 221 [208, 233] versus reproduced 199 (paper 212), McNemar *p* = 0.003; Top-20 246 versus 245 (paper 250). Tied cells are read from intervals, not from a one-structure difference. Generated-versus-refined counts: Supplementary Table 13. **b**, [reserved / to be inserted]. Construction protocol only: Supplementary Note 15, Supplementary Figs. 13, 14. **c**, Laboratory Cu Kα PXRD of LiCoO2. The neutron prefix is simulated from the paired CIF; it is not experimental neutron. The clean model misses every spectroscopic rung. The noise-adapted model with known lattice ranks LiCoO2 2nd from PXRD and 1st from PXRD+ND (no XRF required). Source for ranks: `fig7c_licoo2_ranks.csv`. Remaining collaborator powders: Supplementary Note 14, Supplementary Fig. 12.

# References

1. David, W. I. F. & Shankland, K. Structure determination from powder diffraction data. *Acta Crystallogr. A* **64**, 52–64 (2008).
2. Harris, K. D. M., Tremayne, M. & Kariuki, B. M. Contemporary advances in the use of powder X-ray diffraction for structure determination. *Angew. Chem. Int. Ed.* **40**, 1626–1651 (2001).
3. Sears, V. F. Neutron scattering lengths and cross sections. *Neutron News* **3**, 26–37 (1992).
4. TODO: PXRDGen — Powder diffraction crystal structure determination using generative models. *Nat. Commun.* (2025).
5. TODO: XRDSol — Equivariant diffusion solution for inorganic crystal structure determination from powder X-ray diffraction data.
6. Guo, G. et al. Ab initio structure solutions from nanocrystalline powder diffraction data via diffusion models. *Nat. Mater.* (2025).
7. Li, Q. et al. Experimental powder X-ray diffraction crystal structure determination with RealPXRD-Solver. Preprint at https://arxiv.org/abs/2603.00965 (2026).
8. Antunes, L. M., Butler, K. T. & Grau-Crespo, R. Crystal structure generation with autoregressive large language modeling. *Nat. Commun.* **15**, 10570 (2024).
9. Johansen, F. L. et al. deCIFer: crystal structure prediction from powder diffraction data using autoregressive language models. Preprint at https://arxiv.org/abs/2502.02189 (2025).
10. Uni-3DAR: unified cross-scale 3D generation and understanding via autoregressive modeling. Preprint at https://arxiv.org/abs/2503.16278 (2025).
11. Ong, S. P. et al. Python Materials Genomics (pymatgen): a robust, open-source python library for materials analysis. *Comput. Mater. Sci.* **68**, 314–319 (2013).
12. TODO: neutron calculator as implemented in pymatgen (`NDCalculator`); coherent scattering lengths as tabulated therein, following Sears (ref. 3).
13. Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. *Nat. Mach. Intell.* **5**, 1031–1041 (2023).
14. Batatia, I. et al. MACE: higher order equivariant message passing neural networks for fast and accurate force fields. Preprint at https://arxiv.org/abs/2206.07697 (2022).
15. Toby, B. H. & Von Dreele, R. B. GSAS-II: the genesis of a modern open-source all purpose crystallography software package. *J. Appl. Cryst.* **46**, 544–549 (2013).
16. Davies, D. W. et al. SMACT: semiconducting materials by analogy and chemical theory. *J. Open Source Softw.* **4**, 1361 (2019).
17. Jain, A. et al. Commentary: The Materials Project: a materials genome approach to accelerating materials innovation. *APL Mater.* **1**, 011002 (2013).
18. Kirklin, S. et al. The Open Quantum Materials Database (OQMD). *npj Comput. Mater.* **1**, 15010 (2015).
19. Curtarolo, S. et al. AFLOW: an automatic framework for high-throughput materials discovery. *Comput. Mater. Sci.* **58**, 218–226 (2012).
20. Schmidt, J. et al. Improving machine-learning models in materials science through large datasets. TODO: Alexandria / WBM citations to be completed.
21. Lafuente, B., Downs, R. T., Yang, H. & Stone, N. The power of databases: the RRUFF project. In *Highlights in Mineralogical Crystallography* 1–30 (De Gruyter, 2015).
22. TODO: opXRD / CNRS experimental powder collection as used by RealPXRD-Solver (ref. 7).
