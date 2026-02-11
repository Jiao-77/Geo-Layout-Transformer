# Geo-Layout-Transformer TODOs

本文件汇总项目目标、架构概览、当前完成度与改进计划，按优先级分组并提供可执行清单（复选框）。

## 项目目标（简述）
- 构建用于物理设计版图理解的统一基础模型，面向热点检测、连通性验证、结构匹配等任务。
- 采用“GNN Patch Encoder + 全局 Transformer”的混合架构，支持自监督预训练与任务头微调。

## 架构概览（对应代码位置）
- 数据层：`src/data/`
	- `gds_parser.py`：GDSII/OASIS 解析、按 patch 裁剪与几何特征提取（使用 gdstk）。
	- `graph_constructor.py`：从几何对象构建 PyG 图（节点特征、KNN/Radius 边、元信息）。
	- `dataset.py`：InMemoryDataset 加载处理后的 `.pt` 数据。
- 模型层：`src/models/`
	- `gnn_encoder.py`：可切换 GCN/GraphSAGE/GAT 的 Patch 编码器 + 全局池化。
	- `transformer_core.py`：Transformer 编码器（正余弦位置编码 + EncoderStack）。
	- `task_heads.py`：分类/匹配任务头；`geo_layout_transformer.py` 组装端到端模型。
- 训练与评估：`src/engine/`
	- `trainer.py`：监督训练循环（BCEWithLogitsLoss）；缺少 focal loss 等实现。
	- `evaluator.py`：Accuracy/Precision/Recall/F1/AUC 指标计算。
	- `self_supervised.py`：占位式“掩码版图建模”流程，尚不稳定（见改进项）。
- 脚本与入口：
	- `scripts/preprocess_gds.py`：GDS → 图数据集流水线（保存为 InMemoryDataset）。
	- `scripts/visualize_attention.py`：注意力可视化占位，需实现细节。
	- `main.py`：加载配置、构建数据/模型，并在 pretrain/train/eval 模式下运行。
- 配置：`configs/default.yaml`、`configs/hotspot_detection.yaml`
- 依赖与版本：`pyproject.toml`（Python >=3.12，Torch/PyG 等）；锁文件 `uv.lock`。

## 当前完成度（粗略评估）
- 已完成
	- GDS 解析与 patch 裁剪（含裁剪多边形与面积比例等元信息）。
	- 图构建（节点几何/层特征，KNN/Radius 边，PyG Data 包装）。
	- GNN 编码器（GCN/GraphSAGE/GAT）与 Transformer 主干的基本数据流。
	- 监督训练 Trainer（BCEWithLogitsLoss）、Evaluator 指标管线。
	- 预处理脚本与 InMemoryDataset 持久化；基础日志与配置装载/合并。
	- README 中安装/运行指引（推荐 uv；备选 Conda/Pip）。
- 进行中/占位
	- 自监督预训练（self_supervised）：掩码策略与维度重塑存在假设，需调通与验证。
	- 注意力可视化脚本：仅说明性注释，未接入模型权重与实际权重提取。
	- main.py 数据集切分：目前 train/val 复用同一数据源，留有 TODO。
- 缺失/需改进
	- 任务头与损失的更丰富支持（如 focal loss、class weights、masking/采样）。
	- 训练循环的验证与早停、最佳模型保存、学习率调度等训练工程化能力。
	- 自监督目标的严谨实现（mask 索引与 batch/ptr 对齐、掩码、重建头/投影器）。
	- 可复现实验脚本与最小数据样例；单元测试与快速 CI 校验。
	- CUDA/大图内存管理（梯度累积、混合精度、GraphSAINT/Cluster-GCN 等）。
	- 可观测性（TensorBoard/CSVLogger、随机种子、配置溯源与版本记录）。

## 优先级清单（可执行项）

### P0（立即优先）
- [x] 数据集切分与 DataLoader 管线
	- 在 `main.py` 引入可配置的 train/val/test 切分比例与随机种子；支持从目录/清单载入各 split。
	- 为 `configs/default.yaml` 增加 `splits` 字段；更新 `README*` 用法说明。
- [x] 监督训练工程化
	- 在 `trainer.py` 补充验证阶段与最佳模型保存（`torch.save` 至指定路径）。
	- 引入学习率调度器（如 StepLR/CosineAnnealingWarmRestarts）与早停策略。
	- 支持 class weights/focal loss：在 `trainer.py` 增加 `focal_loss` 实现并在配置选择。
- [x] 自监督预训练修复
	- 明确 batch 内每图的 patch 序列映射：根据 `batch.ptr` 逐图生成 mask 索引，避免跨图混淆。
	- 将掩码作用在输入特征/图结构层而非已池化的图级嵌入；或增加“节点级→patch 聚合→重建头”。
	- 为 `transformer_core` 或单独模块增加重建头（MLP）以回归原 patch 表征；提供单元测试。

### P1（高优）
- [x] 任务头与损失扩展
	- 在 `task_heads.py` 增加多标签分类、回归头；增添可插拔的池化（CLS token/Mean/Max/Attention Pool）。
	- 在 `trainer.py` 支持多任务训练配置（不同 head/loss 的加权）。
- [x] 训练与日志可观测性
	- 增加 TensorBoard/CSVLogger；记录 epoch 指标、学习率、耗时；保存 `config` 与 `git` 提交信息。
	- 固定随机种子（PyTorch/NumPy/环境变量），在 `utils` 中提供 `set_seed()` 并在入口调用。
- [x] 可复现实验与最小数据
	- 提供最小 GDS 示例与对应的 processed `.pt` 小样，便于 CI 与用户快速体验。
	- 在 `scripts/` 增加一键跑通的小样流程脚本（preprocess→train→eval）。

### P2（中优）
- [x] 大图/性能优化
	- 引入混合精度（`torch.cuda.amp`）、梯度累积、可选更小 batch，监控显存。
	- 探索 GraphSAINT/Cluster-GCN 等大图训练策略，并与当前 patch 划分结合。
- [ ] I/O 与生态集成
	- `klayout` Python API 的可选集成与安装脚本说明；解析 OASIS 的路径补全与测试。
	- 在 `graph_constructor.py` 为边策略加入可学习/基于几何关系的拓展（如跨层连接边）。
- [x] 可解释性与可视化
	- 完成 `scripts/visualize_attention.py`：注册 Hook 提取注意力/特征图，绘图并保存到 `docs/`。
	- 在 `Data.node_meta` 基础上支持几何叠加可视化（patch bbox 与局部多边形）。

### P3（后续）
- [ ] 更丰富的自监督任务
	- 对比学习（SimCLR/GraphCL/MaskGIT风格）、上下文预测、旋转/裁剪增广等。
- [ ] 生成式方向探索
	- 以 Transformer 编码为条件，尝试版图片段重建/扩展的生成任务。
- [ ] 文档与示例完善
	- 在 `README*` 增补训练曲线示例、模型结构图与常见问题（FAQ）。

## 风险与边界条件（建议处理）
- 空 patch/稀疏边界：预处理阶段应丢弃无几何或孤立节点过多的 patch，并统计占比。
- 类别不平衡：提供正负样本重采样或损失加权；评估报告中输出混淆矩阵与 PR 曲线。
- 版本与兼容：已将 Python 要求更新为 3.12+；如需老版本 Python，需回溯依赖并测试。
- 随机性：固定随机种子并在日志中写入，以确保结果可复现。

---

维护者可按上述优先级推进，每完成一项请勾选对应复选框并在 PR 中引用本条目以便追踪。

