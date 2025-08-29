from typing import List, Dict, Tuple
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

        node_features: List[List[float]] = []
        node_positions: List[List[float]] = []
        node_layers: List[int] = []
        node_meta: List[Dict] = []

        # 提取每个几何图形的特征（优先使用裁剪后片段的质心；若无裁剪片段，则使用全局质心）
        for geo in geometries:
            layer_idx: int = int(geo["layer"]) if "layer" in geo else 0
            global_bbox = geo.get("global_bbox", None)
            global_points = geo.get("global_points", None)
            clipped_points_list = geo.get("clipped_points_list", []) or []
            clipped_area = float(geo.get("clipped_area", 0.0))
            global_area = float(geo.get("global_area", 0.0))
            area_ratio = float(geo.get("area_ratio", 0.0))
            is_partial = bool(geo.get("is_partial", False))

            # 选择用于节点位置与宽高的几何：若存在裁剪片段，聚合其外接框，否则用全局框
            if clipped_points_list:
                # 合并所有裁剪片段点，计算整体外接框与质心
                all_pts = np.vstack(clipped_points_list)
            elif global_points is not None:
                all_pts = np.array(global_points, dtype=float)
            else:
                # 回退到 bbox 信息（兼容旧格式）
                x_min, y_min, x_max, y_max = geo["bbox"]
                all_pts = np.array([[x_min, y_min], [x_max, y_max]], dtype=float)

            x_min, y_min = np.min(all_pts, axis=0)
            x_max, y_max = np.max(all_pts, axis=0)
            width = float(x_max - x_min)
            height = float(y_max - y_min)
            centroid_x = float(x_min + width / 2.0)
            centroid_y = float(y_min + height / 2.0)

            # 节点特征：质心、宽、高、裁剪面积、全局面积占比、层索引（数值化）
            features = [
                centroid_x,
                centroid_y,
                width,
                height,
                clipped_area,
                (clipped_area / global_area) if global_area > 0 else 0.0,
                float(layer_idx),
                1.0 if is_partial else 0.0,
            ]

            node_features.append(features)
            node_positions.append([centroid_x, centroid_y])
            node_layers.append(layer_idx)
            # 将原始与裁剪的必要元信息保存在 Data 中（以便后续可视化与调试）
            node_meta.append({
                "layer": layer_idx,
                "global_bbox": tuple(global_bbox) if global_bbox is not None else None,
                "global_area": global_area,
                "clipped_area": clipped_area,
                "area_ratio": area_ratio,
                "is_partial": is_partial,
            })

        # 将特征和位置转换为 PyTorch 张量
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(node_positions, dtype=torch.float)

        # 根据选定的策略创建边
        edge_index = self._create_edges(pos)

        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index, pos=pos, y=torch.tensor([label], dtype=torch.float))
        # 附加层索引与元信息（元信息以对象列表形式保存，供上层使用；不会参与张量运算）
        data.layer = torch.tensor(node_layers, dtype=torch.long)
        data.node_meta = node_meta
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
