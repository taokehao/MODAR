# 已锁定的编辑决定（2026-09-12）

用户评估后拍板，下一轮写作必须遵守。评估过程见 `01_narrative_for_review.md`（历史稿，以本文件为准）。

## 脊骨（标题级主张）

粉末衍射单独不可解。同一套自回归晶体语言，把 PXRD、中子衍射和组成探针当成可缺省的条件前缀；在模拟阶梯上，PXRD+ND+XRF 逼近精确化学式；在实验粉末上，同一模型对上专用 PXRD solver 的召回，并在缺公式、缺晶格时仍能按探针降级。

- 不是「我们比 flow solver 更准」。
- 不是「我们发明了晶体语言模型」。
- 是「一个模型、同一表示，既走通多探针信息阶梯，又在实验 PXRD 基准上与专用求解器打平」。

## 创新点嵌套

| 条目 | 正文位置 |
|---|---|
| 晶体语言接上连续谱、覆盖缺模态 | Intro 一句 + Fig. 1 + Methods |
| 自回归快 | Methods 一句引理 + SI 墙钟；不进标题、不进摘要贡献句 |
| 治理过的多模态数据可开源 | Fig. 2 + SI |
| **创新 4（生成/设计）** | **只进 Discussion**；不进 Intro 贡献列表、不进摘要、不进 Results |

Intro 三条科学贡献：(i) 缺模态晶体语言；(ii) 模拟阶梯上的探针互补（含中子配对证据）；(iii) 实验三层：solver 主场（CNRS/RRUFF）、公开 337 无公式梯子、实验室 LiCoO2。337 按 `FIG7_EXPERIMENTAL.md` 写，**不等 formula 档 MACE**。

## 图

| 图 | 决定 |
|---|---|
| Fig. 1–6 | 用户将提供精修版；**以用户交付文件为准**，覆盖仓库里旧的 `paper_figures/figN_*/outputs` |
| Fig. 5 大胞 | **留在正文**，不降 SI |
| Fig. 7 | **三层并一张图**（见 `FIG7_EXPERIMENTAL.md`）：a CNRS/RRUFF vs RPS；b 公开 337 **无公式**光谱梯子；c LiCoO2 |
| Fig. 7b / 337 | **可以写**。光谱档已是 CHGNet+MACE 精修。公式档不用等 MACE（未 refine fusion 作虚线 reference）。旧 8.6% 作废 |
| 合作者 10 例 | **正文只留 LiCoO2 一格**（Fig. 7c）；其余样品、3/10、7/10、库内 9/10 放 SI |

## 本轮写 / 不写

**写：** Abstract（可提 337 的 XRF 开关，不报单一实验准确率）、Introduction、Results 3.1–3.7 含 Fig. 7 三层、Discussion、Methods、captions。

**不写：** 把 337 当 RPS 第二战场；等公式 MACE 才定稿；旧 8.6%；合作者 10 当独立泛化。

## 用语红线

- CNRS：与 RPS 复现统计不可区分（T1 100 vs 101；T20 118 vs 118）。
- RRUFF T1：显著领先（221 vs 199）。RRUFF T20：精修后持平。禁止「全面超过」。
- 必须双报 RPS **论文报告值**与**我们复现值**，差 1–2 个样本用区间说话。
- XRF = 元素质量分数，不是能谱。XPS 不进主图。Fig. 6 noisy ≠ 实验谱。LiCoO2 的 ND 是 CIF 模拟。
- MP-20 前 500 不得与文献全测试集画同一根轴。
