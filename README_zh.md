<div align="center">

<p align="center">
  <img src="docs/images/logo.png" width="240px" alt="Geo-Layout Transformer"/>
</p>

<p>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/stargazers"><img src="https://img.shields.io/github/stars/your-username/Geo-Layout-Transformer.svg" /></a>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/network/members"><img src="https://img.shields.io/github/forks/your-username/Geo-Layout-Transformer.svg" /></a>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/issues"><img src="https://img.shields.io/github/issues-raw/your-username/Geo-Layout-Transformer" /></a>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed-raw/your-username/Geo-Layout-Transformer" /></a>
  <a><img src="https://img.shields.io/badge/python-3.9%2B-blue" /></a>
  <a><img src="https://img.shields.io/badge/PyTorch-2.x-orange" /></a>
</p>

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

</div>

# Geo-Layout Transformer 🚀 🔬

**一个用于物理设计分析的统一、自监督基础模型**

---

## ✨ 亮点 🌟

- **统一基础模型**：覆盖多种物理设计分析任务
- **混合 GNN + Transformer**：从局部到全局建模版图语义
- **自监督预训练**：在无标签 GDSII/OASIS 上学习强泛化表示
- **模块化任务头**：轻松适配（如热点检测、连通性验证）

## 🖥️ 支持系统 💻

- **Python**：3.9+
- **操作系统**：macOS 13+/Apple Silicon、Linux（Ubuntu 20.04/22.04）。Windows 建议使用 **WSL2**
- **深度学习框架**：PyTorch、PyTorch Geometric（CUDA 可选）
- **EDA I/O**：GDSII/OASIS（通过 `klayout` Python API）

## 1. 项目愿景 🎯

**Geo-Layout Transformer** 是一个旨在推动电子设计自动化（EDA）物理设计领域范式转变的研究项目。我们不再依赖于一套零散的、基于启发式规则的工具，而是致力于构建一个统一的基础模型，使其能够理解半导体版图深层次的、上下文相关的“设计语言”。

通过利用新颖的 **图神经网络（GNN）+ Transformer** 混合架构，并在海量未标记的 GDSII 数据上进行预训练，该模型经过微调后，能够出色地完成各种关键的后端分析任务，包括：

*   **高精度连通性验证**：通过理解版图拓扑结构来检测开路和短路。
*   **结构化版图匹配**：实现 IP 复用和设计相似性搜索。
*   **预测性热点检测**：以高准确率和低误报率识别可制造性问题。

我们的愿景是，从目前分散的、任务特定的工具，演进为一个集中的、可复用的“版图理解引擎”，从而加速设计周期，并突破 PPA（功耗、性能、面积）的极限。

## 2. 核心架构 🏗️

该模型的架构设计旨在分层处理版图信息，模仿人类专家从局部细节到全局上下文分析设计的过程。

![架构图](https://i.imgur.com/example.png) <!-- 未来架构图的占位符 -->

1.  **GDSII 到图的处理流水线**：我们将原始的 GDSII/OASIS 文件解析成丰富的异构图表示。每个版图“区块”（Patch）被转换成一个图，其中多边形和通孔是**节点**，它们之间的物理邻接和连通关系是**边**。

2.  **GNN 区块编码器**：一个强大的图神经网络（特指关系图注意力网络 - RGAT）作为“局部规则学习器”。它处理每个区块的图，将复杂的局部几何形状和层间关系编码成一个单一的、丰富的特征向量（嵌入）。这个嵌入向量代表了对该区块的高度语义化总结。

3.  **全局 Transformer 骨干网络**：区块嵌入序列被送入一个 Transformer 模型。至关重要的是，我们注入了**混合二维位置编码**（包括绝对和相对位置），以告知模型每个区块的空间位置。Transformer 的自注意力机制使其能够检测长程依赖关系、重复结构（如标准单元阵列）以及整个芯片的全局上下文模式。

4.  **特定任务头**：从 Transformer 输出的、具有全局上下文感知能力的最终嵌入，被送入简单、轻量级的神经网络“头”（Head）中，以执行特定的下游任务。这种模块化设计使得核心模型能够以最小的代价适应新的应用。

## 🧭 项目结构 📁

```text
Geo-Layout-Transformer/
├─ configs/                  # 训练与任务配置（如 default、hotspot）
├─ scripts/
│  ├─ preprocess_gds.py      # GDSII → 图数据集流水线
│  └─ visualize_attention.py # 注意力与可解释性工具
├─ src/
│  ├─ data/
│  │  ├─ dataset.py          # PyG 数据集/加载器
│  │  ├─ gds_parser.py       # GDS/OASIS 解析
│  │  └─ graph_constructor.py# 异构图构建逻辑
│  ├─ engine/
│  │  ├─ trainer.py          # 训练循环（预训练/微调）
│  │  ├─ evaluator.py        # 评估与指标
│  │  └─ self_supervised.py  # 自监督任务（如掩码版图建模）
│  ├─ models/
│  │  ├─ gnn_encoder.py      # 区块级 GNN 编码器（如 RGAT）
│  │  ├─ transformer_core.py # 全局 Transformer 骨干
│  │  ├─ task_heads.py       # 下游任务头
│  │  └─ geo_layout_transformer.py # 端到端模型组装
│  └─ utils/                 # 配置、日志与通用工具
├─ main.py                   # 入口（预训练/训练/推理）
├─ requirements.txt          # Python 依赖（在 PyTorch/PyG 之后安装）
└─ README*.md                # 中英文文档
```

## 3. 快速上手 ⚙️

### 3.1. 环境要求 🧰

*   Python 3.9+
*   强烈建议使用 Conda 进行环境管理。
*   能够访问 EDA 工具以生成带标签的数据（例如，使用 DRC 工具生成热点标签）。

### 3.2. 安装步骤 🚧

1.  **克隆代码仓库：**
    ```bash
    git clone https://github.com/your-username/Geo-Layout-Transformer.git
    cd Geo-Layout-Transformer
    ```

2.  **创建并激活 Conda 环境：**
    ```bash
    conda create -n geo_trans python=3.9
    conda activate geo_trans
    ```

3.  **安装依赖：**
    本项目需要 PyTorch 和 PyTorch Geometric (PyG)。请根据您的 CUDA 版本遵循官方指南进行安装。

    *   **PyTorch:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   **PyG:** [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

    安装完 PyTorch 和 PyG 后，安装其余的依赖项：
    ```bash
    pip install -r requirements.txt
    ```
    *（注意：您可能需要通过 `klayout` 自身的包管理器或从源码编译来单独安装它，以启用其 Python API）。*

> 提示：GPU 不是必须的。仅 CPU 环境可安装 PyTorch/PyG 的 CPU 版本。

## 4. 项目使用 🛠️

项目的工作流程分为两个主要阶段：数据预处理和模型训练。

### 4.1. 阶段一：数据预处理 🧩

第一步是将您的 GDSII/OASIS 文件转换为模型可以使用的图数据集。

1.  将您的版图文件放入 `data/gds/` 目录。
2.  在 `configs/default.yaml` 中配置预处理参数。您需要定义区块大小、步长、层映射以及图边的构建方式。
3.  运行预处理脚本：
    ```bash
    python scripts/preprocess_gds.py --config-file configs/default.yaml --gds-file data/gds/my_design.gds --output-dir data/processed/my_design/
    ```
    该脚本将解析 GDS 文件，将其划分为多个区块，为每个区块构建一个图，并将处理后的数据保存为 `.pt` 文件以便高效加载。

#### 多边形处理与按区块建图 🧩

在为每个区块（patch）构建图时，我们同时保留多边形的全局信息和区块内（裁剪后）的信息，以稳健处理跨越多个区块的多边形：

- 每个几何对象包含：
  - **全局多边形**：顶点、外接框、面积。
  - **区块内裁剪多边形（可能多个片段）**：顶点、面积，以及 **面积占比**（裁剪/全局）。
  - **is_partial 标记**：指示是否跨区块。
  - **层索引** 与 **区块边界框**。
- 节点特征包含：基于裁剪形状（若无则基于全局）的质心、宽/高、裁剪面积、面积占比、层 id、是否跨区块标志。
- 额外元数据保存在 PyG `Data` 对象中：
  - `data.layer: LongTensor [num_nodes]`
  - `data.node_meta: List[Dict]`，含每个节点的全局/裁剪细节（用于可视化/调试）

该设计借鉴了 LayoutGMN 的结构编码思想，同时与我们现有的 GNN 编码器保持兼容。

### 4.2. 阶段二：模型训练 🏋️

数据集准备就绪后，您就可以开始训练 Geo-Layout Transformer。

#### 自监督预训练（推荐）

为了构建一个强大的基础模型，我们首先在无标签数据上使用“掩码版图建模”任务对其进行预训练。

```bash
python main.py --config-file configs/default.yaml --mode pretrain --data-dir data/processed/my_design/
```
这将训练模型理解物理版图的基本“语法”，而无需任何昂贵的标签。

#### 监督微调

预训练之后，您可以在一个较小的、有标签的数据集上对模型进行微调，以适应像热点检测这样的特定任务。

1.  确保您处理好的数据包含标签。
2.  使用一个特定于任务的配置文件（例如 `hotspot_detection.yaml`），其中定义了模型的任务头和损失函数。
3.  在 `train` 模式下运行主脚本：
    ```bash
    python main.py --config-file configs/hotspot_detection.yaml --mode train --data-dir data/processed/labeled_hotspots/ --checkpoint-path /path/to/pretrained_model.pth
    ```

## 5. 发展路线与贡献 🗺️

这是一个宏伟的项目，我们欢迎任何形式的贡献。我们未来的发展路线图包括：

-   [ ] **更先进的自监督任务**：探索对比学习和其他 SSL 方法。
-   [ ] **模型可解释性**：实现可视化注意力图的工具，以理解模型的决策过程。
-   [ ] **全芯片可扩展性**：集成图分割技术（如 Cluster-GCN）来处理芯片规模的设计。
-   [ ] **生成式设计**：在生成式框架中使用学习到的表示来合成“构建即正确”的版图。

欢迎随时提出 Issue 或提交 Pull Request。

## 致谢 🙏

本项目离不开开源社区的贡献与启发，特别感谢：

- PyTorch 与 PyTorch Geometric，为模型构建与图学习提供可靠基石
- gdstk/klayout，为 GDSII/OASIS 的解析与几何操作提供高效能力
- 科学计算生态（NumPy、SciPy），保障数值计算的稳定性
- 研究工作 LayoutGMN（面向结构相似性的图匹配），启发了我们对多边形/图构建的设计

若您的工作被本项目使用但尚未列出，欢迎提交 Issue 或 PR 以便完善致谢。

---

Made with ❤️ 面向 EDA 研究与开源协作。
