# src/models/geo_layout_transformer.py
import torch
import torch.nn as nn
from .gnn_encoder import GNNEncoder
from .transformer_core import TransformerCore
from .task_heads import ClassificationHead, MultiLabelClassificationHead, RegressionHead, MatchingHead

class GeoLayoutTransformer(nn.Module):
    """完整的 Geo-Layout Transformer 模型。"""

    def __init__(self, config: dict):
        """初始化模型。

        Args:
            config: 包含所有模型超参数的配置字典。
        """
        super(GeoLayoutTransformer, self).__init__()
        self.config = config

        # 1. GNN 编码器：用于将每个版图区块（patch）编码为嵌入向量
        self.gnn_encoder = GNNEncoder(
            node_input_dim=config['model']['gnn']['node_input_dim'],
            hidden_dim=config['model']['gnn']['hidden_dim'],
            output_dim=config['model']['gnn']['output_dim'],
            num_layers=config['model']['gnn']['num_layers'],
            gnn_type=config['model']['gnn']['gnn_type']
        )

        # 2. Transformer 骨干网络：用于捕捉区块之间的全局上下文关系
        self.transformer_core = TransformerCore(
            hidden_dim=config['model']['transformer']['hidden_dim'],
            num_layers=config['model']['transformer']['num_layers'],
            num_heads=config['model']['transformer']['num_heads'],
            dropout=config['model']['transformer']['dropout']
        )

        # 3. 特定于任务的头：根据配置动态创建
        self.task_head = None
        if 'task_head' in config['model']:
            head_config = config['model']['task_head']
            pooling_type = head_config.get('pooling_type', 'mean')
            
            if head_config['type'] == 'classification':
                self.task_head = ClassificationHead(
                    input_dim=head_config['input_dim'],
                    hidden_dim=head_config['hidden_dim'],
                    output_dim=head_config['output_dim'],
                    pooling_type=pooling_type
                )
            elif head_config['type'] == 'multi_label_classification':
                self.task_head = MultiLabelClassificationHead(
                    input_dim=head_config['input_dim'],
                    hidden_dim=head_config['hidden_dim'],
                    output_dim=head_config['output_dim'],
                    pooling_type=pooling_type
                )
            elif head_config['type'] == 'regression':
                self.task_head = RegressionHead(
                    input_dim=head_config['input_dim'],
                    hidden_dim=head_config['hidden_dim'],
                    output_dim=head_config['output_dim'],
                    pooling_type=pooling_type
                )
            elif head_config['type'] == 'matching':
                self.task_head = MatchingHead(
                    input_dim=head_config['input_dim'],
                    output_dim=head_config['output_dim'],
                    pooling_type=pooling_type
                )
            # 可在此处添加其他任务头

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: 一个 PyG 的 Batch 对象，包含了一批次的图数据。

        Returns:
            来自任务头的最终输出张量。
        """
        # 1. 从 GNN 编码器获取区块嵌入
        # PyG 的 DataLoader 会自动将图数据打包成一个大的 Batch 对象
        patch_embeddings = self.gnn_encoder(data)

        # 2. 为 Transformer 重塑形状: [batch_size, seq_len, hidden_dim]
        # 这需要知道批次中每个图包含多少个区块（节点）。
        # 我们可以从 PyG Batch 对象的 `ptr` 属性中获取此信息。
        num_graphs = data.num_graphs
        # `ptr` 记录了每个图的节点数累积和，通过相减得到每个图的节点数
        nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
        # 假设批次内所有图的区块数相同（对于我们的滑动窗口方法是成立的）
        patch_embeddings = patch_embeddings.view(num_graphs, nodes_per_graph[0], -1)

        # 3. 将区块嵌入序列传入 Transformer
        contextual_embeddings = self.transformer_core(patch_embeddings)

        # 4. 将结果传入任务头
        if self.task_head:
            output = self.task_head(contextual_embeddings)
        else:
            # 如果没有定义任务头（例如在自监督预训练中），则返回上下文嵌入
            output = contextual_embeddings

        return output
