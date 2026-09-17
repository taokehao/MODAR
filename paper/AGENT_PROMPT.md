# 给写作 agent 的 prompt（复制下一节即可）

把 `paper/AGENT_PROMPT.md` 里「PROMPT 正文」整段贴进新对话。同时附上精修 Fig. 1–6。337 按 `FIG7_EXPERIMENTAL.md` 写，不等 formula MACE。

---

## PROMPT 正文

```text
你是这篇 Nature Communications 稿的写作 agent。不要重新讨论叙事，不要改脊骨。先读仓库，再按已锁定大纲写英文主文。

## 必读（按顺序）

1. `/home/tkh/python-project/multimodal_csp-v2/paper/LOCKED.md` — 编辑决定，最高优先级
2. `/home/tkh/python-project/multimodal_csp-v2/paper/OUTLINE.md` — 章节与每节主张
3. `/home/tkh/python-project/multimodal_csp-v2/paper/STATUS.md` — 冻结数字与红线
4. `/home/tkh/python-project/multimodal_csp-v2/paper/FIG7_EXPERIMENTAL.md` — Fig. 7 三层与 337 数字（光谱档已 refine，不等 formula MACE）
5. 用户本轮提供的精修 Fig. 1–6（聊天附件，或 `paper/figures_final/`）。图与旧 README 冲突时，以用户的图为准来写 caption 和正文指代（panel a/b/c…）。
6. 需要核对数字时再打开：
   - `paper_figures/fig2_dataset_atlas/README.md`
   - `paper_figures/fig3_multimodal_performance/README.md` 与 `data/figure_3_source_data.csv`
   - `paper_figures/fig4_neutron_value/README.md`
   - `paper_figures/fig5_large_cell/README.md`
   - `paper_figures/fig6_noise_robustness/README.md`
   - `crystal_language/eval_results/chgnet/rps_cnrs126_bootstrap_refined.json`
   - `crystal_language/eval_results/chgnet/rps_rruff269_bootstrap_refined.json`
   - `crystal_language/eval_results/chgnet/ladder_public_si_report_spec_cm.json`
   - `paper_figures/fig7_experimental_pxrd/data/panels/fig7c_licoo2_ranks.csv`
   - `总体实施方案.md` 仅 §8.12（RPS 协议）。不要把实施方案里的训练流水账写进正文。

旧文件 `paper/01_narrative_for_review.md` 是评估过程，已被 LOCKED.md 取代。`paper_figures/si_supplementary/` 里关于公开 337 的 8.6% 等数字全部作废，禁止抄进主文。

## 任务

在 `paper/manuscript/` 写 Nature Communications 风格英文主稿，建议文件：

- `paper/manuscript/main_draft.md`（完整主文：Title, Abstract, Introduction, Results, Discussion, Methods, Figure captions, Acknowledgements 占位）
- `paper/manuscript/figure_captions.md`（Fig. 1–7 图注；7b 用光谱梯子，不要写 reserved）

不要写 LaTeX 除非用户另要。不要重画 Fig. 1–6。不要实现新实验、不要改模型、不要动 checkpoint、不要杀 VASP/ColabFold。不要等公式档 MACE；337 用 spec_cm。

## 脊骨（必须贯穿，不得改写主张）

Powder PXRD is underdetermined. One autoregressive crystal language treats PXRD, neutron diffraction, and composition probes as droppable prefixes. On the simulated ladder, PXRD+ND+XRF approaches an exact-formula reference. On experimental powders, the same model matches a dedicated PXRD solver’s recall and still degrades when formula or lattice is missing.

Not: we are more accurate than a flow solver. Not: we invented crystal language models.

## 本轮范围

写：Abstract、Introduction、Results 3.1–3.7（Fig. 1–7 三层）、Discussion、Methods、captions。

337 按 FIG7_EXPERIMENTAL.md：已知晶格 refined PXRD 0/1、+XRF 70/89、+ND+XRF 75/86。不要等公式 MACE。不要写旧 8.6%。Fig. 7b 不要画成单一实验准确率。

Fig. 5（大胞）留在正文。合作者 10 个样品里正文只写 LiCoO2；3/10、7/10、库内 9/10 若提及，只能放 SI 草稿一句，不要展开成主文泛化。

创新 4（该语言也可用于生成/设计）只出现在 Discussion 末段。不要进 Title、Abstract、Introduction 贡献句、Results。

## 数字（必须与 STATUS.md 一致）

模拟全测试 N=33,944，N_cand=100，rerank：
- lattice-free T1: PXRD 3.8%; +ND 7.8%; +XRF 83.2%; Multi 84.0%; formula 85.9%
- lattice-free T20: Multi 91.5%; formula 93.0%
- known-lattice T1: Multi 90.4%; formula 89.5%
- known-lattice T20: Multi 93.6%; formula 93.9%
Formula is an oracle reference, not a theoretical upper bound.

大胞同一 500 条 >20：specialist Multi vs formula T1 = 66.4% vs 58.4% (free)，72.6% vs 61.0% (known)。

RPS 头对头（refine-before-match，F_chgnet）：
- CNRS n=126: T1 100 [91, 109] vs repro 101 (reported 98); T20 118 vs 118 (reported 116)
- RRUFF n=269: T1 221 [208, 233] vs 199 (reported 212), McNemar p=0.003; T20 246 vs 245 (reported 250)
Wording: CNRS statistically indistinguishable from reproduced RPS; RRUFF T1 significantly ahead; RRUFF T20 tied after refine. Never “outperforms RPS overall”. Always print both reported and reproduced RPS numbers.

LiCoO2: laboratory Cu Kα PXRD; ND is CIF-simulated. Clean model misses all spectroscopic rungs. noise-adapted + known lattice: PXRD rank 2 → PXRD+ND Top-1. Do not call this experimental neutron.

数据：unified 10,207,687 → Tier-A 3,524,185; train 3,456,202; test 33,944.

## 表述红线

- 不写 first crystal LM / not limited to graphs 作为创新。
- 不把 speed 当贡献；最多 Methods 一句 + 指向 SI（4090 D, median 19 s/structure at N=100）。
- XRF in the main ladder is exact elemental mass fractions, not a measured XRF spectrum.
- XPS stays out of main figures; near-zero increment is SI.
- Fig. 6 “noisy” = synthetic artefacts, not measured patterns.
- MP-20: if mentioned, SI only; never plot our first-500 83.2% on the same axis as PXRDGen full 9046.
- Do not invent panel contents that are not in the user-supplied Fig. 1–6.
- Do not overwrite `crystal_language/runs/tierA_s2_di/checkpoint_last.pt`.

## 文风

Nature Communications 方法文：短 Intro，Results 由图驱动，先结论后数字，限制写清楚。用 “formula reference” 而非 “theoretical upper bound”。Tied 用置信区间，不用 1 个样本的差。中文仓库注释不要写进英文稿。作者/单位/基金用 TODO。

写完后在 `paper/manuscript/main_draft.md` 文首列：写了哪些节、337 用的是 spec_cm 而不是公式 MACE。不要承诺去等那 5 个 shard。
```
