# FlowMM 思路与进度总结

> **[2026-06-10 冻结]** 本路线已停止开发，代码移至 `archive/flowmm`（DiffCSP 在 `archive/diffcsp`）。
> 项目主线已铆定为自回归晶体语言模型，见根目录 `总体实施方案.md`。本文档仅作历史记录，
> 文中 `external_models/flowmm` 等相对路径已不再有效。

本文档总结当前项目从 Uni-3DAR 转向 FlowMM / crystal-native 生成路线后的总体思路、代码改造、数据适配、训练进度、已知问题和后续计划。

## 1. 为什么转向 FlowMM

项目最初使用 Uni-3DAR 做晶体结构生成，但早期 MP+WBM 小模型结果不理想。核心问题不只是模型规模，而是晶体结构本身具有平移不变性、周期性以及旋转相关的对称性质，而 Uni-3DAR 更接近固定 3D 网格 / token 化表示，天然不具备这些结构归纳偏置。

之前为了让 Uni-3DAR 可训练，我们采用了晶体标准化、唯一化、几何中心和质心对齐等处理。但这会带来一个反直觉现象：在 lattice-conditioned 条件下，模型可能把结构随机旋转，结果反而比 lattice-free 更差。这说明单纯在数据预处理层面“固定姿态”不能可靠解决晶体生成的对称性问题。

因此当前路线改为使用更 crystal-native 的方法：

- 用晶格参数、分数坐标、原子类型作为原生表示；
- 在周期边界条件下构图，而不是固定 3D voxel；
- 采用 FlowMM / Riemannian Flow Matching 作为主线；
- 保留 DiffCSP / CDVAE 等作为可比较或可借鉴的晶体生成基线；
- 通过条件生成和消融实验系统评估 PXRD、中子散射、XRF、XPS 价态、元素集合、精确化学式和 lattice 信息的贡献。

## 2. 当前 FlowMM 训练目标

当前优先目标不是立刻训练全部 1000 万级统一数据，而是先用 MP+WBM 全量数据打通 FlowMM 条件生成主流程。MP+WBM 数据规模约 40 万条，已经包含增强后的 PXRD、中子散射、XPS 价态等信息，适合作为中等规模验证集。

当前主要训练设置：

- 数据：`data_prep/outputs/official_mp_wbm_multimodal`
- 训练集：`train.lmdb`
- 验证集：`valid.lmdb`
- 测试集：`test.lmdb`
- 模型配置：`model=conditional_null_params`
- 数据配置：`data=pxrd_mof_mp_wbm_lmdb_lattice_conditioned`
- 条件集合：`pxrd_neutron_xrf_xps`
- 训练模式：lattice-conditioned
- GPU：单节点 4 张 4090D
- W&B project：`pxrd_mof_flowmm`

当前正在运行的正式任务：

- Slurm job：`798`
- 状态：运行中
- 开始时间：`2026-06-04 13:52:29`
- 当前进度：约 `Epoch 102 / 2000`
- 当前 batch 进度：约 `2422 / 6231`
- W&B run：`https://wandb.ai/aimslab200/pxrd_mof_flowmm/runs/g18w1k0b`
- 粗略速度：一个 epoch 约 `16-17 分钟`
- 若完整跑 2000 epoch，预计约 `23 天`
- 当前 sbatch time limit：`7 天`，所以该任务预计会在约 `epoch 600` 左右被时间限制中止，除非后续手动停止或调整策略。

## 3. 数据适配思路

### 3.1 不重新下载数据

项目现有数据已经下载并增强完成，包括：

- MP+WBM：约 410,011 条；
- Alexandria：约 5,929,013 条 raw/enhanced；
- OQMD：约 1,407,387 条 enhanced；
- AFLOW：约 2,528,949 条 enhanced；
- 统一 LMDB：train 约 10,003,856 条，valid 约 101,978 条，test 约 101,853 条。

因此当前工作不再重复下载数据，而是围绕已有 LMDB / JSONL 数据格式做 FlowMM 适配。

### 3.2 直接 LMDB adapter

为了优先打通 MP+WBM 训练，已实现直接读取 MP+WBM LMDB 的 FlowMM 数据集：

```text
external_models/flowmm/remote/DiffCSP-official/diffcsp/pl_data/dataset.py
```

新增或改造的核心类：

```text
CrystalNativeLmdbDataset
```

它直接读取：

```text
data_prep/outputs/official_mp_wbm_multimodal/{train,valid,test}.lmdb
```

并输出 FlowMM / DiffCSP 可用的 PyG `Data` 对象，包括：

- `frac_coords`
- `atom_types`
- `lengths`
- `angles`
- `edge_index`
- `to_jimages`
- `num_atoms`
- `num_nodes`
- `y`
- 条件字段，如 PXRD、中子散射、XRF、元素集合、化学式计数、价态向量等。

### 3.3 图构建缓存

FlowMM 训练前需要把晶体结构构造成图。这个过程使用 CrystalNN / Voronoi 等方法，CPU 开销很大。为避免每次训练都重新构图，数据集会把构图结果缓存为 `.pt` 文件。

当前缓存命名包含条件、lattice 设置、最大原子数和 max_records：

```text
train_<condition_set>_<use_lattice_condition>_max_atoms_<max_atoms>_max_records_<max_records>.pt
```

例如全量 MP+WBM lattice-conditioned + PXRD/neutron/XRF/XPS 的缓存为：

```text
train_pxrd_neutron_xrf_xps_True_max_atoms_128_max_records_null.pt
valid_pxrd_neutron_xrf_xps_True_max_atoms_128_max_records_null.pt
test_pxrd_neutron_xrf_xps_True_max_atoms_128_max_records_null.pt
```

这样可以避免小样本 smoke test 的缓存被误用于全量训练。

### 3.4 坏结构跳过与记录

全量 MP+WBM 中存在少数结构在 CrystalNN / Voronoi 构图时失败，例如：

```text
ValueError: No Voronoi neighbors found for site - try increasing cutoff
```

之前这种单条坏结构会中断整个预处理。现在已改为：

- 构图失败时跳过该结构；
- 记录到 `*.bad_ids.jsonl`；
- 写入 `*.bad_summary.json`；
- 继续处理后续样本。

目前在旧的未过滤缓存中曾发现 2 条构图失败结构，例如：

```text
mp:mp-1247813, AlAg7Se5
mp:mp-994911, S
```

### 3.5 大结构过滤

MP+WBM 中存在超过 128 atoms 的结构，最大曾统计到 444 atoms。FlowMM 当前配置为：

```yaml
max_atoms: 128
```

旧 adapter 没有真正执行这个过滤，导致训练中遇到超大结构后触发 `torch_scatter` 的 CUDA device-side assert。现在已在 `CrystalNativeLmdbDataset` 中加入 `max_atoms` 过滤：

- 大于 `max_atoms` 的结构不会进入缓存；
- 会被记录为 `TooManyAtoms`；
- 其信息会进入 bad ids / summary。

这一步是当前训练能稳定进入 epoch 的关键修复之一。

## 4. 条件生成设计

本项目目标不是简单无条件生成晶体，而是围绕实验表征信息做条件生成，并通过消融实验评估每类条件的贡献。

当前支持的条件集合：

```text
pxrd
pxrd_element
pxrd_xrf
pxrd_neutron
pxrd_neutron_xrf
pxrd_neutron_xrf_xps
pxrd_formula
```

这些条件对应后续推理 / 消融实验：

```text
PXRD only
PXRD + element set
PXRD + XRF
PXRD + neutron
PXRD + neutron + XRF
PXRD + neutron + XRF + XPS-valence
PXRD + exact formula
```

目标是观察逐步加入信息后，生成质量是否逐渐接近 `PXRD + exact formula` 的理论上限。

## 5. 条件向量编码

当前第一版条件编码是一个轻量图级条件向量，维度为 492：

```text
5 * 2 + 119 * 4 + 6 = 492
```

组成如下：

- PXRD summary：5 维；
- neutron summary：5 维；
- element set vector：119 维；
- XRF mass fraction：119 维；
- formula count vector：119 维；
- XPS valence vector：119 维；
- lattice parameters：6 维。

其中 PXRD / neutron 目前不是完整序列编码器，而是先用 summary 统计量：

- mean
- std
- max
- sum
- length / normalization factor

这是一个保守的 first-pass 方案，目的是先打通训练与消融框架。后续可以替换为真正的谱图编码器，例如 1D CNN、Transformer、Fourier features 或 peak-based encoder。

## 6. FlowMM 网络改造

主要改动文件：

```text
external_models/flowmm/src/flowmm/model/arch.py
external_models/flowmm/src/flowmm/model/model_pl.py
external_models/flowmm/scripts_model/conf/vectorfield/rfm_cspnet.yaml
external_models/flowmm/scripts_model/conf/model/conditional_null_params.yaml
```

### 6.1 CSPNet 条件注入

在 CSPNet 中新增：

- `condition_dim`
- `condition_embedding`
- `graph_conditions` 输入

图级条件先经过 MLP 映射到 hidden_dim，再 broadcast 到该图的所有节点，加入节点特征：

```text
node_features = node_features + condition_embedding(graph_conditions)
```

这样模型可以在预测坐标 flow / lattice flow 时使用多模态条件。

### 6.2 vectorfield 配置

在：

```text
external_models/flowmm/scripts_model/conf/vectorfield/rfm_cspnet.yaml
```

加入：

```yaml
condition_dim: ${model.condition_dim}
```

使模型配置能控制是否启用条件输入。

### 6.3 条件模型配置

新增：

```text
external_models/flowmm/scripts_model/conf/model/conditional_null_params.yaml
```

当前内容：

```yaml
defaults:
  - null_params

use_conditions: true
condition_dim: 492
condition_set: ${data.condition_set}
use_lattice_condition: ${data.use_lattice_condition}
use_affine_stats: false
```

`use_affine_stats: false` 是因为 MP+WBM 的项目专用 lattice affine stats 尚未计算。当前先绕过 FlowMM 原有数据集统计依赖。

## 7. lattice-free / lattice-conditioned 设计

项目明确需要比较：

- lattice-free：模型同时生成 lattice 和 fractional coordinates；
- lattice-conditioned：给定真实 lattice，仅生成 fractional coordinates。

### 7.1 lattice-conditioned

当：

```yaml
use_lattice_condition: true
```

当前训练逻辑为：

- lattice 参数进入条件向量；
- 初始点 `x0` 的 lattice 部分被替换为目标 lattice；
- 目标 flow 的 lattice 分量置零；
- 预测 flow 的 lattice 分量置零；
- lattice loss 置零。

直观上，lattice-conditioned 模式只学习在给定晶胞下生成分数坐标。

### 7.2 lattice-free

当：

```yaml
use_lattice_condition: false
```

模型需要同时学习：

- atom / type 相关部分，当前 `conditional_null_params` 中 atom types 近似视为已知或不作为主要生成目标；
- fractional coordinates flow；
- lattice flow。

lattice-free 是后续与 lattice-conditioned 对比的关键设置。

## 8. 当前训练任务记录

### 8.1 job 789

状态：失败。

原因：

- 数据预构建已经完成；
- 进入训练后在早期 batch 触发 `torch_scatter` CUDA device-side assert；
- 后续定位为旧缓存中包含超过模型 `max_atoms` 假设的大结构。

### 8.2 job 791

状态：完成，但结果无效。

原因：

- 没有同步到 W&B；
- 更重要的是误用了 smoke test 产生的 512 条小缓存；
- 因此虽然任务完成，但不是全量 MP+WBM 训练。

修复：

- cache suffix 加入 `max_records`；
- sbatch 默认强制 W&B online，不继承本地 `WANDB_MODE=disabled`；
- 删除误用的小缓存；
- 重新提交正式任务。

### 8.3 job 797

状态：被取消。

现象：

- 停在 cache 预构建阶段；
- 没有进入训练；
- 没有产生 W&B run。

### 8.4 job 798

状态：当前正式运行中。

关键信息：

```text
JobID: 798
Node: 4090-Cluster
Config: data=pxrd_mof_mp_wbm_lmdb_lattice_conditioned model=conditional_null_params
Condition: pxrd_neutron_xrf_xps
Batch: train=16 val=32 test=32
W&B: https://wandb.ai/aimslab200/pxrd_mof_flowmm/runs/g18w1k0b
```

当前已确认：

- W&B online 正常；
- 4 卡 DDP 正常；
- 全量过滤缓存已构建完成；
- 训练已经进入 epoch；
- 当前约 `Epoch 102 / 2000`。

粗略耗时：

- 一个 epoch 约 `16-17 分钟`；
- 完整 2000 epoch 约需 `23 天`；
- 当前 sbatch 限时 `7 天`；
- 因此该任务预计无法自然跑完 2000 epoch，大概率在约 `epoch 600` 左右被 Slurm time limit 截断。

## 9. sbatch 脚本

当前训练脚本：

```text
scripts/submit_train_flowmm_mp_wbm.sbatch
```

主要设置：

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4090D:4
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00
```

默认配置：

```bash
DATA_CONFIG=pxrd_mof_mp_wbm_lmdb_lattice_conditioned
MODEL_CONFIG=conditional_null_params
CONDITION_SET=pxrd_neutron_xrf_xps
BATCH_SIZE_TRAIN=16
BATCH_SIZE_VAL=32
BATCH_SIZE_TEST=32
NUM_WORKERS=8
MAX_EPOCHS=2000
WANDB_PROJECT=pxrd_mof_flowmm
WANDB_MODE=online
```

脚本逻辑：

1. 激活 `flowmm` Conda 环境；
2. 设置 `PROJECT_ROOT` 和数据路径；
3. 检查 `WANDB_API_KEY`；
4. 单进程预构建 dataloader cache，避免 DDP 多进程同时写缓存；
5. 使用 `srun python -u scripts_model/run.py ...` 启动 4 卡 DDP 训练；
6. 日志写入：

```text
logs/flowmm_mp_wbm_<job_id>.out
logs/flowmm_mp_wbm_<job_id>.err
logs/flowmm_mp_wbm_<job_id>.log
```

## 10. 当前已知问题

### 10.1 训练时间过长

当前 `max_epochs=2000`，但按当前速度完整训练要约 23 天，而 Slurm 限时只有 7 天。

后续可选策略：

- 降低 `MAX_EPOCHS` 到更实际的范围，例如 300-600；
- 保持 2000，但确保 checkpoint resume 流程稳定；
- 先看 W&B 上 100-300 epoch 的 loss 趋势，再决定是否继续长训；
- 调大 batch size 或优化 dataloader，但要先评估显存和吞吐。

### 10.2 tqdm 日志过大

`logs/flowmm_mp_wbm_798.out` 和 `.log` 会被 tqdm 进度条刷得很大，目前已经数百 MB。

后续建议：

- 禁用或降低 tqdm refresh；
- 只保留 rank 0 日志；
- 配置 Lightning progress bar；
- 避免使用 `tee` 记录重复日志；
- 后续更多依赖 W&B 查看训练曲线，而不是读本地超大日志。

### 10.3 条件编码仍然粗糙

当前 PXRD / neutron 只使用 summary 特征，并没有真正学习谱图形状。

后续建议：

- 为 PXRD / neutron 增加 1D CNN encoder；
- 或加入 peak list / peak embedding；
- 或用 Transformer / Perceiver 处理谱图序列；
- 加入 condition dropout，让一个模型支持不同条件缺失场景；
- 对不同 condition_set 做统一训练或多模型对照。

### 10.4 lattice stats 尚未项目化

当前 MP+WBM 暂时复用 FlowMM 原有 `mp_20` dataset key，并在 conditional config 中设置：

```yaml
use_affine_stats: false
```

后续正式训练统一数据前应计算项目专用 lattice stats。

### 10.5 统一大数据集尚未正式接入训练

当前只打通了 MP+WBM。下一步若迁移到 1000 万级统一 LMDB，需要考虑：

- cache 文件极大；
- 构图耗时极长；
- 单个 `.pt` 全量缓存不可扩展；
- 需要 shard / lazy loading / LMDB cache；
- 数据源不均衡问题；
- 多源验证与评估拆分策略。

## 11. 后续建议路线

### 11.1 先观察 job 798

短期建议继续观察 job `798` 的 W&B 曲线：

- train loss 是否稳定下降；
- val/loss 是否下降或平台；
- lattice-conditioned 下 `loss_l=0` 是否符合预期；
- `loss_f` 是否是主要优化项；
- 是否存在明显过拟合或发散。

如果 200-300 epoch 后曲线已经稳定，可以考虑提前停止，进入下一组实验。

### 11.2 跑 lattice-free 对照

当前 798 是 lattice-conditioned。下一步应提交对应的 lattice-free 训练：

```text
data=pxrd_mof_mp_wbm_lmdb_lattice_free
model=conditional_null_params
data.condition_set=pxrd_neutron_xrf_xps
```

目标是重新检查最初 Uni-3DAR 中的异常现象是否仍存在：

- lattice-conditioned 是否优于 lattice-free；
- 若不是，问题是在模型、条件、数据，还是评估流程。

### 11.3 做条件消融训练 / 推理

按照以下 ladder：

```text
PXRD only
PXRD + element set
PXRD + XRF
PXRD + neutron
PXRD + neutron + XRF
PXRD + neutron + XRF + XPS-valence
PXRD + exact formula
```

可以有两种路线：

1. 多模型训练：每个 condition_set 训练一个模型；
2. 单模型 + condition dropout：训练一个支持条件缺失的模型，推理时遮蔽不同条件。

当前代码已经支持通过 `data.condition_set` 控制输入条件，但 condition dropout 还没有系统实现。

### 11.4 改进谱图 encoder

当前 summary encoder 主要用于打通流程。为了真正利用 PXRD / neutron，应增加更强的谱图编码。

可选实现顺序：

1. 1D CNN 编码 PXRD / neutron；
2. 加 peak detection 后使用 peak list encoder；
3. 对谱图加入随机噪声、峰位偏移、强度扰动增强；
4. 将谱图 encoder 输出和元素 / formula / lattice 条件融合。

### 11.5 迁移到统一大数据

在 MP+WBM 跑通后，再迁移到：

```text
data_prep/outputs/official_unified_multimodal
```

需要优先解决：

- 分片缓存；
- 构图预处理并行化；
- 坏结构记录；
- 数据源采样权重；
- 训练恢复机制；
- W&B run 命名和实验矩阵管理。

## 12. 当前结论

FlowMM 方向已经完成从“理论选择”到“可运行训练任务”的关键跨越。现在已经有：

- 可读取现有 MP+WBM LMDB 的条件 dataloader；
- 可跳过坏结构并记录 bad ids 的预处理；
- 可区分 lattice-free / lattice-conditioned 的训练逻辑；
- 可接收 PXRD / neutron / XRF / XPS / formula / lattice 条件的 CSPNet；
- 可在 4 卡 Slurm 上运行并同步 W&B 的训练脚本；
- 正在运行的正式 MP+WBM lattice-conditioned 任务 `798`。

但目前仍处于第一版工程验证阶段。当前模型的条件编码还比较轻量，训练时间配置也偏长。下一阶段应围绕 W&B 曲线判断是否提前停止 798，然后启动 lattice-free 对照和条件消融实验，并逐步把 summary 条件编码升级为真正的谱图 encoder。
