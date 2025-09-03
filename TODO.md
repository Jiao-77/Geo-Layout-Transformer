# TODO — Geo-Layout-Transformer 🚀

目的：遍历项目并把发现的未实现/待完善项整理到此文件，方便后续开发分配与跟踪。📝

简短项目说明 ✨
- 这是一个面向半导体版图（GDSII/OASIS）的研究型工程，目标构建一个混合 GNN + Transformer 的“版图理解”基础模型（自监督预训练 + 任务微调），用于热点评估、连通性校验、版图匹配等下游任务。

检索与覆盖说明（工具扫描结果） 🔎
- 已扫描主要入口和核心模块：`README*.md`, `main.py`, `src/models/*`, `src/data/*`, `src/engine/*`, `scripts/*`。
- 发现显式未实现/占位符（`pass`、`TODO`）的位置列在下方。

-一览：显式未实现 / 需要实现（按优先级排序）

- [ ] 1) 必要：数据处理与加载（高优先级） ⚠️
- 文件：`src/data/dataset.py`
  - 问题：继承 `InMemoryDataset` 的 `download()` 和 `process()` 方法均为 `pass`。
  - 影响：无法自动将原始 GDS/OASIS 转换并打包为 PyG 可加载的 `data.pt`。`main.py` 依赖 `LayoutDataset(root=...)` 加载数据，会在没有 `processed` 数据时失败。
  - 建议实现：
    1. `download()`：可选，从远程或指定路径复制原始文件（若不需要可留空并在 README 标注）。
    2. `process()`：读取 `raw_dir` 下已由 `scripts/preprocess_gds.py` 生成的 `.pt` 或中间文件，或直接在此处调用解析与图构建逻辑（调用 `src/data/gds_parser.py` 和 `src/data/graph_constructor.py`），最后保存 `torch.save((data, slices), self.processed_paths[0])`。
    3. 文档化输入目录结构和所需文件名约定。
  - 估时：3–8 小时，取决于是否复用 `scripts/preprocess_gds.py`。

- [ ] 2) 必要：预处理脚本（高优先级） 🔧
- 文件：`scripts/preprocess_gds.py`
  - 问题：脚本中存在 `pass` 和 `TODO`（未实现从标签文件加载标签或完整的预处理流程）。
  - 影响：无法从 GDS/OASIS 生成可训练的数据集（patch 切分、polygon 裁剪、节点/边构建、保存为 `.pt`）。
  - 建议实现：
    1. 实现或封装 `gds_parser`（基于 `gdstk` 或 `klayout`）以读取多层几何并输出 polygon 列表与层信息。
    2. 实现 patch 切分（窗口大小、stride）、polygon 裁剪与 `is_partial`、area ratio 计算。
    3. 调用 `graph_constructor` 构造 PyG `Data`（节点特征、边、metadata），并保存为单个或批量 `.pt` 文件放入 `processed_dir`。
    4. 提供 `--overwrite`、`--workers`、`--verbose` 等 CLI 参数。
  - 依赖：`gdstk` 或 `klayout`（README 中已提及）。
  - 估时：2–16 小时（实现完整解析 + 并行化视复杂度而定）。

- [ ] 3) 必要：训练脚本中的数据集划分与 checkpoint（中/高优先级） 🗂️
- 文件：`main.py`
  - 问题：存在 TODO，当前将整个 `LayoutDataset` 直接用于 train/val loaders 而非划分；模型 checkpoint 加载被注释（示例中注释掉 `load_state_dict`）。
  - 影响：无法做标准的训练/验证/测试分割，也缺少断点重载逻辑。
  - 建议实现：
    1. 在 `main.py` 中实现基于 `random_split` 或按设计文件/布局分层划分（确保跨-layout 的分割策略），并将结果保存 `splits/` 以保证可复现性。
    2. 实现 checkpoint 的保存（按 epoch/metric）和加载逻辑（支持 optimizer 和 scheduler state）。
  - 估时：1–3 小时。

- [ ] 4) 必要/需修正：模型中批次/序列维度处理（中优先级） 🧩
- 文件：`src/models/geo_layout_transformer.py`
  - 问题：代码里直接用 `nodes_per_graph[0]` 假设每个图（sample）包含相同数量的 patch（nodes），然后用 `.view(num_graphs, nodes_per_graph[0], -1)` 强制 reshape。这在真实数据里通常不成立（patch 数量/节点数会变化）。
  - 影响：当样本 patch 数不同或数据使用不定长序列时会崩溃或产生错误的上下文分割。
  - 建议实现：
    1. 使用 `torch_geometric` 的 `Batch` 提供的信息按-图聚合 patch embeddings（例如，对每个图做 mean/max pooling，或构建 padded sequences 并 mask）。
    2. 另外可在 `graph_constructor` 处保证每个样本序列长度固定（但这限制较大）。
  - 估时：2–6 小时。

- [ ] 5) 必要/改进：Trainer 功能不完整（中优先级） 🚦
- 文件：`src/engine/trainer.py`
  - 问题：仅支持少数优化器和 BCE 损失；没有早停、学习率调度、checkpoint 保存、验证调用；示例中注释掉了 Evaluator 的使用。
  - 影响：难以进行标准训练流程与调参。
  - 建议实现：
    1. 增加 checkpoint 保存/加载（model + optimizer + epoch）。
    2. 支持 scheduler（如 CosineAnnealingLR、ReduceLROnPlateau）与早停逻辑。
    3. 在 `run()` 中每个 epoch 后调用 `Evaluator`（或传入回调）做验证与模型选择。
    4. 扩展损失函数注册，添加 `cross_entropy`、`focal`、`dice`（视任务而定）。
  - 估时：3–8 小时。

- [ ] 6) 改进/增强：任务头与可扩展性（中优先级） 🧠
- 文件：`src/models/task_heads.py` 与 `src/models/geo_layout_transformer.py`
  - 问题：任务头目前仅示例了 classification 与 matching，两者接口可能需要标准化（输入 shape、masking、loss 约定）。
  - 建议：定义统一的 Head 接口（forward 接受 embeddings + mask，可返回 logits + aux），并在配置（configs/*.yaml）中声明 head 类型与损失配置。
  - 估时：2–4 小时。

- [ ] 7) 可选：scripts/visualize_attention.py 与可解释性工具（低优先级） 🔍
- 说明：README 提到 attention 可视化，但 `scripts/visualize_attention.py` 需要检查是否完整实现（未详细扫描）。如果目标是可解释性，应实现从 Transformer attention 到版图坐标/多边形映射的工具链。
- 估时：4–12 小时（视可视化深度）。

- [ ] 8) 项目文档 & CI（低优先级） 📚
- 问题：`pyproject.toml` 中 Python 要求为 3.12（但 README 写 3.9+），依赖列表为空，`requirements.txt` 存在但需与 `pyproject` 同步。
- 建议：统一 python 版本约定、完善 `pyproject.toml` dependencies 或使用 `requirements.txt`，添加 basic `tox`/`github actions` 用于 lint/test。
- 估时：1–3 小时。

隐含的设计改进（建议） 💡
- 增加端到端的单元/集成测试（最小例：人工构造的 patch -> graph -> forward pass），确保 pipeline 各步正确。
- 在 `scripts/preprocess_gds.py` 中加入小样本模式（debug 用），能快速构造少量样本用于单元测试。
- 考虑在 `src/data/` 中添加一个轻量的 synthetic generator（随机几何与层），便于 CI 下的快速运行和回归测试。

建议的短期工作分配（建议先做 1→2→3） 🔜
- A. 实现 `scripts/preprocess_gds.py`（若已有成熟解析器可复用）：2–16h
- B. 实现 `src/data/dataset.py::process()` 加载 `processed/` 数据并写入 `data.pt`：3h
- C. 修复 `geo_layout_transformer` 中的序列 reshape（改为按图聚合或 padding+mask）：3–6h
- D. 在 `main.py` 中实现数据划分与 checkpoint load/save：1–3h
- E. 在 `trainer` 中加入 eval & checkpoint：3–6h

- 质量门（quality gates） ✅
- 在完成 A+B 后，应跑通一个最小端到端 smoke test：生成 1–5 个 processed `.pt`，用 `main.py --mode pretrain` 或 `--mode train` 在 1 epoch 上跑通（CPU 可行）。
- 增加 2 个单元测试：
  1. graph_constructor 测试：输入简单 polygon 输出节点/边及元数据
  2. model 前向测试：使用 synthetic batch 验证 forward 不崩溃并返回期望 shape

文件与代码位置清单（已发现的占位实现）
- src/data/dataset.py — download(), process(): pass
- scripts/preprocess_gds.py — 主要预处理逻辑存在 pass/TODO
- main.py — TODO: 数据集划分；checkpoint 加载示例被注释
- src/models/geo_layout_transformer.py — 假定每个图拥有相同 patch 数（reshape 问题），建议改为可变长度处理
- src/engine/trainer.py — 基础训练循环已实现，但缺少评估调用、checkpoint、scheduler 支持

后续步骤（我可以为你做的）
- 如果你同意，我将按优先级 **实现/修复 A + B + C**：
  1. 实现 `scripts/preprocess_gds.py` 的基础版本（支持 gdstk 作为依赖）并保存 `processed/*.pt`。
  2. 在 `src/data/dataset.py` 中实现 `process()` 以加载 `processed` 文件并生成 `data.pt`。
  3. 修复 `geo_layout_transformer` 中的 reshape，采用 pooling 或 padded sequences + mask。
  4. 运行一个快速 smoke test（1 epoch，CPU），并把结果写入 `TODO.md` 下的进度条目。

要求与注意事项（请确认或提供）
- 你希望我直接修改代码并在仓库中提交这些改动吗？（我已准备好直接修改并运行 smoke test）。
- 如果有偏好的 GDS 解析库（`gdstk` 或 `klayout`），请说明；我会优先使用 `gdstk`（requirements.txt 已列出）。

需求覆盖映射
- "遍历项目代码，找出项目做什么" —— Done（在文档与简短项目说明中覆盖）。
- "找出哪些没有实现的地方" —— Done（列出显式 `pass` / `TODO`，并补充潜在的设计缺陷与改进项）。
- "整理到 TODO.md 里面" —— Done（此文件即为输出）。

变更记录
- 创建：`TODO.md`（列出问题与修复建议）。

最后简短说明：如果你允许我继续实现优先级 A+B+C，我将开始具体编码并在每个重要阶段给出进度更新（包括运行结果和短时间的 smoke tests）。
