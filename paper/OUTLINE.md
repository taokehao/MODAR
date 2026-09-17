# 本轮正文大纲（已锁定）

目标期刊：Nature Communications（Collection: AI-empowered X-ray and neutron studies）。英文主文。Methods 置后。约 4500–6000 词。七张主图。

工作标题（可微调，不改主张）：
**Complementary X-ray and neutron probes for crystal-structure solution in one autoregressive language model**

## Abstract（~150–180 词，四拍）

1. 粉末 PXRD 欠定；中子与组成探针提供互补衬度。
2. 晶体写成可解析语言，谱为条件前缀，一个模型覆盖缺模态。
3. 模拟全测试 PXRD+ND+XRF 逼近精确化学式 reference（无晶格 T1 84.0% vs 85.9%；已知晶格 90.4% vs 89.5%）。
4. 同一协议在 CNRS-126 / RRUFF-269 上与开源 PXRD solver 打平（RRUFF Top-1 领先）。公开 337 条实测 Cu Kα 上，无化学式时 PXRD-only 接近零、加上 XRF 后 Top-1 到 70/337；实验室 LiCoO2 上噪声适配 + 模拟 ND 将正确结构从第 2 推到 Top-1。

不要出现生成设计、首次 LM、比扩散快、单一「实验准确率」。

## Introduction

- 粉末反演为何欠定（相邻 Z、轻原子 / Li / H）。
- 相关工作三层：PXRD-CSP（PXRDGen, XRDSol, PXRDnet, RealPXRD-Solver）；晶体序列模型（CrystaLLM, deCIFer, Uni-3DAR）；无人把 ND 接到同一生成模型作为独立物理通道。
- 贡献 (i)(ii)(iii) 含 337 无公式梯子。创新 4 不出现。
- 导向 Fig. 1。

## Results

### 3.1 缺模态晶体语言（Fig. 1）

CIF 风语法、几何 bin、公式约束解码；PXRD/ND 栅格化 prefix（不进 token 流）；XRF 为质量分数。一个权重覆盖缺模态与已知/未知晶格。速度只作为「故可每条采 N=100」的一句。

### 3.2 治理过的语料（Fig. 2）

Funnel 10.2M → 3.52M Tier-A；train 3,456,202 / valid 34,039 / test 33,944。卖点是泄漏审计与统一模拟物理，不是源库名单。

### 3.3 信息阶梯（Fig. 3）

N=33,944，N_cand=100，rerank。组成探针把问题变成可解；PXRD-only 无晶格 T1 3.8%；+ND 7.8%；+XRF 83.2%；Multi 84.0% vs formula 85.9%。已知晶格 Multi 90.4% 略高于 formula 89.5%，须写明 formula 是 reference 不是理论上限。T20：无晶格 Multi 91.5% vs formula 93.0%；已知 93.6% vs 93.9%。XPS 不出现。

### 3.4 中子在何处工作（Fig. 4）

10k 对齐子集，配对 bootstrap。PXRD→+ND 正增益。ND-given-XRF：lattice-free CI 过零（如实写）；known-lattice 显著。晶系热图为描述性点估计。这是 Collection 的物理心脏。

### 3.5 大胞（Fig. 5）——正文保留

全测试 ≤20 vs >20。同 500 条 >20：specialist Multi 无晶格 T1 66.4% vs formula 58.4%；已知 72.6% vs 61.0%。Routing 是容量分配，不是新架构。

### 3.6 合成赝象（Fig. 6）

N=500。noisy = 合成赝象。clean 权重在噪声输入上成分条件崩溃；noisy-FT 是部署前提。deploy 匹配仅 noisy + lattice-free；Rietveld-lite 改排序不改大胞召回。

### 3.7 实验粉末（Fig. 7）

细节与数字以 `FIG7_EXPERIMENTAL.md` 为准。

**7a CNRS/RRUFF。** PXRD+formula+已知晶格，完整 refine。与 RPS 打平，RRUFF T1 领先。

**7b 公开 337。** 无化学式。已知晶格 refined：PXRD 0/1，+XRF **70/89**，+ND+XRF 75/86。PXRD→XRF 是唯一显著的已知晶格步。公式 MACE 不等。ND/XRF 为 CIF 模拟。

**7c LiCoO2。** 实验室 PXRD；ND 模拟。Clean miss；noisy-exp 已知晶格 PXRD 第 2 → +ND Top-1。

## Discussion

- 离散 token ≈ 拓扑；冻晶格 MLIP ≈ 度量精度。
- 失败模式：无组成的实验 PXRD、大胞召回、实验 ND 域差、XPS 无增量。
- **创新 4 只在这里：** 同一语言与约束解码也是条件生成 / 设计的低成本载体；本文实证停在反演，未做材料发现。
- 不承诺 6M 重训、GRPO、坐标 flow。

## Methods（置后）

表示与词表；谱 prefix；训练（论文 mix checkpoint，勿改权重）；解码与独立集温度；物理 rerank；refine-before-match；StructureMatcher `ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True`；RPS 复现；公开 337 SI 重编码（原胞、轴/波长重映射到 Cu Kα）；LiCoO2 数据来源。337 光谱档 refine = CHGNet∪MACE；公式档正文不引用未完成的 MACE。

## SI

MP-20 口径与泄漏；速度；XPS 负结果；合作者其余样品与 novelty audit；温度选择独立集；试过无效的杠杆；337 无晶格全表与 n≤20。旧 SI docx 里的 337 数字全部作废，不要抄。
