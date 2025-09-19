<!-- README.md -->
<div align="center">

<p align="center">
  <img src="docs/images/logo.png" width="240px" alt="Geo-Layout Transformer"/>
</p>

<p>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/stargazers"><img src="https://img.shields.io/github/stars/your-username/Geo-Layout-Transformer.svg" /></a>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/network/members"><img src="https://img.shields.io/github/forks/your-username/Geo-Layout-Transformer.svg" /></a>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/issues"><img src="https://img.shields.io/github/issues-raw/your-username/Geo-Layout-Transformer" /></a>
  <a href="https://github.com/your-username/Geo-Layout-Transformer/issues?q=is%3Aissue+is%3Aclosed"><img src="https://img.shields.io/github/issues-closed-raw/your-username/Geo-Layout-Transformer" /></a>
  <a><img src="https://img.shields.io/badge/python-3.12%2B-blue" /></a>
  <a><img src="https://img.shields.io/badge/PyTorch-2.x-orange" /></a>
</p>

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

</div>

# Geo-Layout Transformer 🚀 🔬

**A Unified, Self-Supervised Foundation Model for Physical Design Analysis**

---

## ✨ Highlights

- **Unified foundation model** for diverse physical design analysis tasks
- **Hybrid GNN + Transformer** architecture capturing local-to-global layout semantics
- **Self-supervised pretraining** on unlabeled GDSII/OASIS for strong transferability
- **Modular task heads** for easy adaptation (e.g., hotspot detection, connectivity)

## 🖥️ Supported Systems

- **Python**: 3.12+
- **OS**: macOS 13+/Apple Silicon, Linux (Ubuntu 20.04/22.04). Windows via **WSL2** recommended
- **Frameworks**: PyTorch, PyTorch Geometric (with CUDA optional)
- **EDA I/O**: GDSII/OASIS (via `klayout` Python API)

## 1. Vision 🎯

The **Geo-Layout Transformer** is a research project aimed at creating a paradigm shift in Electronic Design Automation (EDA) for physical design. Instead of relying on a fragmented set of heuristic-based tools, we are building a single, unified foundation model that understands the deep, contextual "language" of semiconductor layouts.

By leveraging a novel hybrid **Graph Neural Network (GNN) + Transformer** architecture and pre-training on massive amounts of unlabeled GDSII data, this model can be fine-tuned to excel at a variety of critical back-end analysis tasks, including:

*   **High-Precision Connectivity Verification**: Detecting opens and shorts by understanding the layout topology.
*   **Structural Layout Matching**: Enabling IP reuse and design similarity search.
*   **Predictive Hotspot Detection**: Identifying manufacturability issues with high accuracy and low false positives.

Our vision is to move from disparate, task-specific tools to a centralized, reusable "Layout Understanding Engine" that accelerates the design cycle and pushes the boundaries of PPA (Power, Performance, and Area).

## 2. Core Architecture 🏗️

The model's architecture is designed to hierarchically process layout information, mimicking how a human expert analyzes a design from local details to global context.

![Architecture Diagram](https://i.imgur.com/example.png)  <!-- Placeholder for a future architecture diagram -->

1.  **GDSII to Graph Pipeline**: We parse raw GDSII/OASIS files into a rich, heterogeneous graph representation. Each layout "patch" is converted into a graph where polygons and vias are **nodes**, and their physical adjacencies and connectivity are **edges**.

2.  **GNN Patch Encoder**: A powerful Graph Neural Network (specifically, a Relational Graph Attention Network - RGAT) acts as a "local rule learner". It processes the graph of each patch, encoding the complex local geometries and inter-layer relationships into a single, rich feature vector (embedding). This embedding represents a high-level semantic summary of the patch.

3.  **Global Transformer Backbone**: The sequence of patch embeddings is fed into a Transformer model. Crucially, we inject **hybrid 2D positional encodings** (both absolute and relative) to inform the model of each patch's spatial location. The Transformer's self-attention mechanism allows it to detect long-range dependencies, repetitive structures (like standard cell arrays), and global contextual patterns across the entire chip.

4.  **Task-Specific Heads**: The final, context-aware embeddings from the Transformer are fed into simple, lightweight neural network "heads" for specific downstream tasks. This modular design allows the core model to be adapted to new applications with minimal effort.

## 🧭 Project Structure

```text
Geo-Layout-Transformer/
├─ configs/                  # Training & task configs (e.g., default, hotspot)
├─ scripts/
│  ├─ preprocess_gds.py      # GDSII → graph dataset pipeline
│  └─ visualize_attention.py # Attention/interpretability utilities
├─ src/
│  ├─ data/
│  │  ├─ dataset.py          # PyTorch Geometric dataset/dataloader
│  │  ├─ gds_parser.py       # GDS/OASIS parsing helpers
│  │  └─ graph_constructor.py# Hetero-graph construction logic
│  ├─ engine/
│  │  ├─ trainer.py          # Train loop (pretrain/fine-tune)
│  │  ├─ evaluator.py        # Evaluation & metrics
│  │  └─ self_supervised.py  # SSL tasks (e.g., masked layout modeling)
│  ├─ models/
│  │  ├─ gnn_encoder.py      # Patch-level GNN encoder (e.g., RGAT)
│  │  ├─ transformer_core.py # Global Transformer backbone
│  │  ├─ task_heads.py       # Downstream task heads
│  │  └─ geo_layout_transformer.py # End-to-end model composition
│  └─ utils/                 # Config, logging, misc utilities
├─ main.py                   # Entry point (pretrain/train/infer)
├─ requirements.txt          # Python deps (install after PyTorch/PyG)
└─ README*.md                # English/Chinese documentation
```

## 3. Getting Started ⚙️

### 3.1. Prerequisites 🧰

*   Python 3.12+
*   Dependency management: using uv is recommended for fast, reproducible installs (uv.lock provided). Conda/Python is supported as an alternative.
*   Access to EDA tools for generating labeled data (e.g., a DRC engine for hotspot labels).

### 3.2. Installation 🚧

#### A) Using uv (recommended)

1) Install uv (one-time):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Clone the repository:

```bash
git clone https://github.com/your-username/Geo-Layout-Transformer.git
cd Geo-Layout-Transformer
```

3) Ensure Python 3.12 is available (uv can manage it):

```bash
uv python install 3.12
```

4) Create the environment and install dependencies from uv.lock/pyproject:

```bash
uv sync
```

Notes:
- For CUDA builds of PyTorch/PyG, follow the official installers first, then install the rest via uv:
  - PyTorch: https://pytorch.org/get-started/locally/
  - PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  After installing the correct Torch/PyG wheels, you may run `uv sync --frozen` to install the remaining packages.
- You may need to install `klayout` separately (package manager or from source) to enable its Python API.

#### B) Using Python/Conda (alternative)

1) Clone the repository:

```bash
git clone https://github.com/your-username/Geo-Layout-Transformer.git
cd Geo-Layout-Transformer
```

2) Create and activate an environment (Conda example):

```bash
conda create -n geo_trans python=3.12
conda activate geo_trans
```

3) Install PyTorch and PyTorch Geometric per your CUDA setup:

- PyTorch: https://pytorch.org/get-started/locally/
- PyG: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

4) Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

> Tip: GPU is optional. For CPU-only environments, install the CPU variants of PyTorch/PyG.
> Note: You may need to install `klayout` separately to enable its Python API.

## 4. Project Usage 🛠️

The project workflow is divided into two main stages: data preprocessing and model training.

### 4.1. Stage 1: Data Preprocessing 🧩

The first step is to convert your GDSII/OASIS files into a graph dataset that the model can consume.

1.  Place your layout files in the `data/gds/` directory.
2.  Configure the preprocessing parameters in `configs/default.yaml`. You will need to define patch size, stride, layer mappings, and how to construct graph edges.
3.  Run the preprocessing script:
    - Using uv (recommended):
      ```bash
      uv run python scripts/preprocess_gds.py --config-file configs/default.yaml --gds-file data/gds/my_design.gds --output-dir data/processed/my_design/
      ```
    - Using Python/Conda:
      ```bash
      python scripts/preprocess_gds.py --config-file configs/default.yaml --gds-file data/gds/my_design.gds --output-dir data/processed/my_design/
      ```
    This script will parse the GDS file, divide it into patches, construct a graph for each patch, and save the processed data as `.pt` files for efficient loading.

#### Polygon handling and per-patch graphs 🧩

When building a graph for each patch, we now preserve both global and per-patch (clipped) polygon information to robustly handle polygons spanning multiple patches:

- Each geometry retains:
  - **Global polygon**: vertices, bbox, area.
  - **Clipped polygon(s)** in the patch: vertices (may be multiple fragments), area, and the **area ratio** (clipped/global).
  - **is_partial** flag indicating cross-patch polygons.
  - **Layer index** and the **patch bbox**.
- Node features include centroid, width/height from clipped shape (or global if no clip), clipped area, area ratio, layer id, and partial flag.
- Extra metadata is attached on the PyG `Data` object:
  - `data.layer: LongTensor [num_nodes]`
  - `data.node_meta: List[Dict]` with per-node global/clipped details (for visualization/debugging)

This follows the spirit of LayoutGMN’s structural encoding while staying compatible with our GNN encoder.

### 4.2. Stage 2: Model Training 🏋️

Once the dataset is ready, you can train the Geo-Layout Transformer.

#### Self-Supervised Pre-training (Recommended) ⚡

To build a powerful foundation model, we first pre-train it on unlabeled data using a "Masked Layout Modeling" task.

```bash
# Using uv (recommended)
uv run python main.py --config-file configs/default.yaml --mode pretrain --data-dir data/processed/my_design/

# Using Python/Conda
python main.py --config-file configs/default.yaml --mode pretrain --data-dir data/processed/my_design/
```
This will train the model to understand the fundamental "grammar" of physical layouts without requiring any expensive labels.

#### Supervised Fine-tuning 🎯

After pre-training, you can fine-tune the model on a smaller, labeled dataset for a specific task like hotspot detection.

1.  Ensure your processed data includes labels.
2.  Use a task-specific config file (e.g., `hotspot_detection.yaml`) that defines the model head and loss function.
3.  Run the main script in `train` mode:
  ```bash
  # Using uv (recommended)
  uv run python main.py --config-file configs/hotspot_detection.yaml --mode train --data-dir data/processed/labeled_hotspots/ --checkpoint-path /path/to/pretrained_model.pth

  # Using Python/Conda
  python main.py --config-file configs/hotspot_detection.yaml --mode train --data-dir data/processed/labeled_hotspots/ --checkpoint-path /path/to/pretrained_model.pth
  ```

## 5. Roadmap & Contribution 🗺️

This project is ambitious and we welcome contributions. Our future roadmap includes:

-   [ ] **Advanced Self-Supervised Tasks**: Exploring contrastive learning and other SSL methods.
-   [ ] **Model Interpretability**: Implementing tools to visualize attention maps to understand the model's decisions.
-   [ ] **Full-Chip Scalability**: Integrating graph partitioning techniques (e.g., Cluster-GCN) to handle chip-scale designs.
-   [ ] **Generative Design**: Using the learned representations in a generative framework to synthesize "correct-by-construction" layouts.

Please feel free to open an issue or submit a pull request.

## Acknowledgments 🙏

We stand on the shoulders of open-source communities. This project draws inspiration and/or utilities from:

- PyTorch and PyTorch Geometric for model building and graph learning
- gdstk/klayout for GDSII/OASIS parsing and geometry operations
- Scientific Python stack (NumPy, SciPy) for numerical robustness
- Research works such as LayoutGMN (graph matching for structural similarity) that informed our polygon/graph handling design

If your work is used and not listed here, please open an issue or PR so we can properly credit you.
