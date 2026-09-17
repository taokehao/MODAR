# Fig. 7 实验验证：重设计（2026-09-17）

不等那 5 个 **formula 档 MACE** shard。它们不回答 337 的科学问题。

## 为什么可以不等

337 要回答的不是「再打一场 RPS」。RPS 头对头已经在 CNRS/RRUFF 上用完整 CHGNet+MACE 精修做完了。337 要回答的是：

> 没有化学式时，多探针梯子在**实测 Cu Kα PXRD** 上还在不在？

卡住的 5 个 shard 全是 `pxrd_formula` 的大胞 MACE。光谱档（PXRD / +ND / +XRF / +ND+XRF / +XPS）的 CHGNet **和** MACE 都已 11/11。公式档只是 reference，不是方法。

本轮 337 主数字：光谱档 = 与 CNRS 相同的 refine-before-match（CHGNet ∪ MACE）；公式档若出现，只用 **未 refine fusion** 或 CHGNet-only，并标明不是完整 MACE。

产物：`eval_results/chgnet/ladder_public_si_report_spec_cm.json`（光谱 merge 在 `ladder_public_si/*_refined_spec/`，不覆盖调度器将来的 `_refined_cm`）。

## 三层各讲一句，不要抢同一句

| 层 | 条件 | 主张 | 不是 |
|---|---|---|---|
| **a CNRS/RRUFF** | PXRD + 化学式 + 已知晶格 | 同一 AR 模型在专用 solver 主场打平（RRUFF T1 领先） | 不是多模态证明 |
| **b 公开 337** | 实测 PXRD ± 模拟 ND/XRF，**无化学式** | 组成探针是实验谱上的开关；PXRD-only ≈ 0 是适用范围 | 不是「实验准确率 x%」；不是再赛 RPS |
| **c LiCoO2** | 实验室 RAW + 模拟 ND | 噪声适配后，ND 把正确结构从第 2 推到 Top-1 | 不是 n=10 的百分比；不是实验中子 |

顺序不能反：先证明在 PXRD+formula 主场站得住，再证明拿掉公式后梯子仍由 XRF 驱动，最后用一个实验室案例把 ND 说成人话。

## Panel 画法

**a** 分组柱：CNRS / RRUFF × T1/T20。三根柱：我们 refined、RPS 复现、RPS 报告值。误差棒用我们的配对 bootstrap。图注：tied 由区间论证。

**b** 337 **计数**（不要画成 0–100% 准确率轴）。主系列 = 已知晶格 refined Top-1 / Top-20：

| 条件 | T1 | T20 | oracle（池内） |
|---|---:|---:|---:|
| PXRD | 0 | 1 | 2 |
| +ND | 0 | 17 | 19 |
| +XRF | **70** [56, 85] | **89** [73, 105] | 92 |
| +ND+XRF | **75** [60, 90] | **86** | 88 |

公式 reference（未 refine fusion，虚线或浅色）：T1 73 / T20 86。一句：光谱 +XRF 的 Top-1（70）已经贴近公式档未 refine 的 73，所以 **等公式 MACE 只是在抬 reference 天花板，不改梯子故事**。

配对步（已知晶格，refined）：

- PXRD → +XRF：T1 **+70 [+56, +85]**（唯一必须画的步）
- +XRF → +ND+XRF：T1 **+5 [−2, +12]**，过零 → 已知晶格上 ND 不是加分项（与模拟 10k 一致）
- 无晶格 +XRF → +ND+XRF：T1 **+15 [+6, +24]** → 写在 panel b 的一句 annotation 或 SI，不要为了它去等 MACE

XPS 不进 7b。n≤20（122 条）可作空心点：+XRF 53/67，+ND+XRF 59/65。

**c** LiCoO2：实验 PXRD vs 模拟 ND；clean 全 miss；noisy-exp + 已知晶格 PXRD 第 2 → +ND Top-1。

## 正文用语

- 337 的 ND/XRF 是 CIF 模拟；实测的只有 Cu Kα。
- 不报 70/337 = 21% 当「实验准确率」。报计数，并对照 PXRD-only = 0。
- 不把 337 和 CNRS 的 100/126 横比当失败：条件不同（有无公式、胞大小、谱–CIF 失配）。
- 旧公开集 8.6%（另一编码、另一模型）作废。

## 那 5 个 MACE shard

可以停：它们只服务公式 reference 的 MACE 精修，正文 7b 不依赖。留着只对 SI「公式档 refine 到和 CNRS 同一后端」有用，墙钟可能再数周。未停的话不要让写作再等。

## 已出图（2026-09-17，光谱档 CHGNet ∪ MACE）

- 主图：`paper_figures/fig7_experimental_pxrd/outputs/figure_7_experimental_pxrd.{png,pdf,svg}`
- SI：S17 无晶格梯子、S18 n≤20 拆分、S19 配对增量、S20 XPS 增量
  `paper_figures/si_supplementary/outputs/fig_s17_public337_lattice_free.*`
  `paper_figures/si_supplementary/outputs/fig_s18_public337_size_split.*`
  `paper_figures/si_supplementary/outputs/fig_s19_public337_paired_steps.*`
  `paper_figures/si_supplementary/outputs/fig_s20_public337_xps.*`

