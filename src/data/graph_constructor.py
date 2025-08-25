from typing import List, Dict
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree
import numpy as np

class GraphConstructor:
    """从几何图形列表中构建 PyTorch Geometric 的 Data 对象（即图）。"""

    def __init__(self, edge_strategy: str = "knn", knn_k: int = 8, radius_d: float = 1.0):
        """
        Args:
            edge_strategy: 创建边的策略（'knn' 或 'radius'）。
            knn_k: KNN 策略中的 K（最近邻的数量）。
            radius_d: 半径图策略中的半径大小。
        """
        self.edge_strategy = edge_strategy
        self.knn_k = knn_k
        self.radius_d = radius_d

    def construct_graph(self, geometries: List[Dict], label: int = 0) -> Data:
        """为单个区块构建一个图。

        Args:
            geometries: 来自 GDSParser 的几何图形字典列表。
            label: 图的标签（例如，0 表示非热点，1 表示热点）。

        Returns:
            一个 PyTorch Geometric 的 Data 对象。
        """
        # 如果没有几何图形，则返回 None
        if not geometries:
            return None

        node_features = []
        node_positions = []
        # 提取每个几何图形的特征
        for geo in geometries:
            x_min, y_min, x_max, y_max = geo["bbox"]
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            centroid_x = x_min + width / 2
            centroid_y = y_min + height / 2

            # 特征包括：中心点坐标、宽度、高度、面积
            features = [centroid_x, centroid_y, width, height, area]
            node_features.append(features)
            node_positions.append([centroid_x, centroid_y])

        # 将特征和位置转换为 PyTorch 张量
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(node_positions, dtype=torch.float)

        # 根据选定的策略创建边
        edge_index = self._create_edges(pos)

        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index, pos=pos, y=torch.tensor([label], dtype=torch.float))
        return data

    def _create_edges(self, node_positions: torch.Tensor) -> torch.Tensor:
        """根据选定的策略创建边。"""
        nodes_np = node_positions.numpy()
        if self.edge_strategy == "knn":
            # 使用 cKDTree 进行高效的 K 最近邻搜索
            tree = cKDTree(nodes_np)
            # 查询每个点的 k+1 个最近邻（包括自身）
            dist, ind = tree.query(nodes_np, k=self.knn_k + 1)
            # 创建边列表，排除自环
            row = np.repeat(np.arange(len(nodes_np)), self.knn_k)
            col = ind[:, 1:].flatten()
            edge_index = torch.tensor([row, col], dtype=torch.long)

        elif self.edge_strategy == "radius":
            # 使用 cKDTree 查找在指定半径内的所有点对
            tree = cKDTree(nodes_np)
            pairs = tree.query_pairs(r=self.radius_d)
            edge_index = torch.tensor(list(pairs), dtype=torch.long).t().contiguous()
        else:
            raise ValueError(f"未知的边构建策略: {self.edge_strategy}")

        return edge_index
