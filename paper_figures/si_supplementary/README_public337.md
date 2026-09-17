# Public-337 SI figures (S17–S20)

These panels sit behind manuscript Fig. 7b. They use the spectroscopic
refine-before-match pools (`CHGNet ∪ MACE`) in
`crystal_language/eval_results/chgnet/ladder_public_si_report_spec_cm.json`.
They do **not** wait for the leftover formula-MACE shards.

```bash
python paper_figures/si_supplementary/prepare_public337.py
python paper_figures/si_supplementary/plot_public337_si.py
```

| Figure | Claim |
|---|---|
| **S17** | Lattice-free ladder, including XPS as an SI-only last rung |
| **S18** | Known-lattice Top-1 on all 337 versus the n_sites ≤ 20 subset (122) |
| **S19** | Paired bootstrap increments (XRF is the switch; ND helps only lattice-free) |
| **S20** | XPS after ND+XRF does not move Top-1 outside the ND+XRF interval |

ND/XRF/XPS on this set are CIF-simulated. Counts are structure matches, not an
accuracy rate.
