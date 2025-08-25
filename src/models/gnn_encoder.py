import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool

class GNNEncoder(nn.Module):
    """基于 GNN 的编码器，用于生成区块（Patch）的嵌入向量。"""

    def __init__(self, node_input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, gnn_type: str = 'gcn'):
        """
        Args:
            node_input_dim: 输入节点特征的维度。
            hidden_dim: 隐藏层的维度。
            output_dim: 输出区块嵌入向量的维度。
            num_layers: GNN 层的数量。
            gnn_type: 使用的 GNN 层类型（'gcn', 'graphsage', 'gat'）。
        """
        super(GNNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(self.get_gnn_layer(node_input_dim, hidden_dim, gnn_type))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(self.get_gnn_layer(hidden_dim, hidden_dim, gnn_type))
        
        # 输出层
        self.layers.append(self.get_gnn_layer(hidden_dim, output_dim, gnn_type))

        # 读出函数，用于将节点嵌入聚合为图级别的嵌入
        self.readout = global_mean_pool

    def get_gnn_layer(self, in_channels, out_channels, gnn_type):
        """根据类型获取 GNN 层。"""
        if gnn_type == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif gnn_type == 'graphsage':
            return SAGEConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            # 注意：GATConv 可能需要额外的参数，如 heads
            return GATConv(in_channels, out_channels)
        else:
            raise ValueError(f"不支持的 GNN 类型: {gnn_type}")

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: 一个 PyTorch Geometric 的 Data 或 Batch 对象。

        Returns:
            一个代表区块的图级别嵌入的张量。
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 通过所有 GNN 层
        for layer in self.layers:
            x = layer(x, edge_index)
            x = torch.relu(x)

        # 全局池化以获得图级别的嵌入
        graph_embedding = self.readout(x, batch)
        return graph_embedding
