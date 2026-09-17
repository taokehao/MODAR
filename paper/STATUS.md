# 写作冻结表（2026-09-12）

原则：定稿数字可进正文。337 光谱档已可写（不等 formula MACE）。见 `FIG7_EXPERIMENTAL.md`。

## 已冻结，本轮可写

| 块 | 口径 | 主数字 | 来源 |
|---|---|---|---|
| 模拟阶梯 | Tier-A test，N=33,944，N_cand=100，rerank | 无晶格 Multi T1 **84.0%** / T20 **91.5%**；formula T1 85.9% / T20 93.0%。已知晶格 Multi T1 **90.4%** / T20 **93.6%**；formula T1 89.5% / T20 93.9%。PXRD-only 无晶格 T1 3.8%；+ND 7.8%；+XRF 83.2% | `crystal_language/eval_results/ladder_tierA_s2_full_neutron`；Fig. 3 CSV |
| 中子配对 | 对齐 10k，配对 bootstrap | PXRD→+ND 正增益；ND-given-XRF 无晶格 CI 过零、已知晶格显著 | `ladder_tierA_s2_10k/stats` |
| 大胞 | 同 500 条 >20 | specialist Multi 无晶格 T1 **66.4%** vs formula **58.4%**；已知 **72.6%** vs **61.0%** | `bigcell_ab/oracle_same_subset_comparison.json` |
| 合成噪声 | N=500 / 配置；noisy ≠ 实验 | noisy-FT 是可用性前提；deploy 匹配只用于 noisy+lattice-free | Fig. 6 数据包 |
| CNRS / RRUFF vs RPS | mix `tierA_s2_di`，T=1.0/k50，N=100，F_chgnet，refine-before-match | CNRS T1 **100** [91, 109] vs repro 101（报告 98）；T20 **118** vs 118（报告 116）。RRUFF T1 **221** [208, 233] vs 199（报告 212），p=0.003；T20 **246** vs 245（报告 250） | `eval_results/chgnet/rps_cnrs126_bootstrap_refined.json`，`rps_rruff269_bootstrap_refined.json` |
| 数据治理 | Funnel + split | 10,207,687 → 3,524,185 Tier-A；train 3,456,202 / valid 34,039 / test 33,944 | Fig. 2 |
| 速度（SI） | 4090 D，N=100 | 中位 **19.0 s / 结构** | `paper_figures/si_supplementary/data/runtime_summary.json` |
| LiCoO2 | noisy-exp，已知晶格 | clean 全 miss；noisy：PXRD best_k=2，PXRD+ND **T1**；ND 为 CIF 模拟 | `paper_figures/fig7_experimental_pxrd/data/panels/fig7c_licoo2_ranks.csv` |

## 337 光谱档（2026-09-17 已可写）

已知晶格 CHGNet∪MACE refine（`ladder_public_si_report_spec_cm.json`）：PXRD 0/1；+XRF **70/89**；+ND 0/17；+ND+XRF **75/86**。无晶格 +XRF 18/24，+ND+XRF **33/48**（ND-given-XRF T1 +15 [+6, +24]）。公式档 MACE 未齐：正文只用未 refine fusion（已知 73/86）作 reference。

## 本轮禁止

| 块 | 原因 |
|---|---|
| 等公式 MACE 才画 Fig. 7b | 光谱梯子已完整；公式只是虚线 reference |
| 旧 337 的 8.6% | 另一编码/模型，作废 |
| 合作者 10 的 3/10、7/10 | 正文只留 LiCoO2 |
| XPS 主文卖点 | 全模拟近零增量 |

## 红线（任何一轮都成立）

- 不写首次晶体 LM；不把「快」当标题贡献。
- 实验 ND：LiCoO2 / 337 图上的 ND 是 CIF 模拟。
- 对 RPS 必须双报报告值与复现值；tied 用 CI，禁止全面超过。
- 不覆盖 `crystal_language/runs/tierA_s2_di/checkpoint_last.pt`（mtime 必须保持 2026-08-26 10:11）。
- 不杀 VASP / ColabFold。那 5 个 formula MACE shard **可以停**（见 `FIG7_EXPERIMENTAL.md`）；未停也不要让写作再等它们。
