# src/data/gds_parser.py
from typing import List, Dict, Tuple
import gdstk
import numpy as np

class GDSParser:
    """解析 GDSII/OASIS 文件，提取指定区块内的版图几何图形。"""

    def __init__(self, gds_file: str, layer_mapping: Dict[str, int]):
        """初始化 GDSParser。

        Args:
            gds_file: GDSII/OASIS 文件的路径。
            layer_mapping: 一个字典，将 GDS 的层/数据类型字符串（例如 "1/0"）映射到整数索引。
        """
        self.gds_file = gds_file
        self.layer_mapping = layer_mapping
        # 使用 gdstk 读取 GDS 文件
        self.library = gdstk.read_gds(gds_file)
        # 获取顶层单元
        self.top_cell = self.library.top_level()[0]

    def get_patches(self, patch_size: float, patch_stride: float) -> List[Tuple[float, float, float, float]]:
        """生成覆盖整个版图的区块坐标。

        Args:
            patch_size: 正方形区块的尺寸（单位：微米）。
            patch_stride: 滑动窗口的步长（单位：微米）。

        Returns:
            一个包含所有区块边界框 (x_min, y_min, x_max, y_max) 的列表。
        """
        # 获取顶层单元的边界框
        x_min, y_min, x_max, y_max = self.top_cell.bb()
        patches = []
        # 使用步长在 x 和 y 方向上生成区块
        for x in np.arange(x_min, x_max, patch_stride):
            for y in np.arange(y_min, y_max, patch_stride):
                patches.append((x, y, x + patch_size, y + patch_size))
        return patches

    def extract_geometries_from_patch(self, patch_bbox: Tuple[float, float, float, float]) -> List[Dict]:
        """从给定的区块中提取所有几何对象，并记录全局与区块内（裁剪后）的信息。

        说明：
        - 为了处理跨越多个区块的多边形，本函数会计算多边形与区块边界框的布尔相交，
          得到位于该区块内的裁剪多边形，并同时记录原始（全局）多边形信息。

        Args:
            patch_bbox: 区块的边界框 (x_min, y_min, x_max, y_max)。

        Returns:
            一个字典列表，每个字典代表一个几何对象及其属性：
            - global_points: 原始多边形顶点（Nx2 ndarray）
            - global_bbox: 原始多边形边界框
            - global_area: 原始多边形面积
            - clipped_points: 与区块相交后的裁剪多边形顶点（Mx2 ndarray，可能为空）
            - clipped_area: 裁剪后面积（可能为 0）
            - area_ratio: 裁剪面积 / 原始面积（用于衡量跨区块比例）
            - is_partial: 是否为跨区块（裁剪面积 < 原始面积）
            - layer: 层映射到的整数索引
            - patch_bbox: 当前区块边界框
        """
        x_min, y_min, x_max, y_max = patch_bbox
        rect = gdstk.rectangle(x_min, y_min, x_max, y_max)
        polygons = self.top_cell.get_polygons(by_spec=True)
        geometries: List[Dict] = []

        for (layer, datatype), poly_list in polygons.items():
            layer_str = f"{layer}/{datatype}"
            if layer_str not in self.layer_mapping:
                continue
            layer_idx = self.layer_mapping[layer_str]

            for poly in poly_list:
                p_xmin, p_ymin, p_xmax, p_ymax = poly.bb()
                # 快速边界框测试（若无相交则跳过）
                if p_xmax < x_min or p_xmin > x_max or p_ymax < y_min or p_ymin > y_max:
                    continue

                # 全局多边形点与面积
                global_points = np.array(poly.points, dtype=float)
                global_area = abs(gdstk.Polygon(global_points).area())

                # 与区块矩形做相交，可能返回多个多边形
                clipped = gdstk.boolean([poly], [rect], "and", precision=1e-3, layer=layer, datatype=datatype)
                clipped_points_list: List[np.ndarray] = []
                clipped_area = 0.0
                if clipped:
                    for cpoly in clipped:
                        pts = np.array(cpoly.points, dtype=float)
                        if pts.size == 0:
                            continue
                        area = abs(gdstk.Polygon(pts).area())
                        if area <= 0:
                            continue
                        clipped_points_list.append(pts)
                        clipped_area += area

                area_ratio = (clipped_area / global_area) if global_area > 0 else 0.0
                is_partial = area_ratio < 0.999  # 允许微小数值误差

                geometries.append({
                    "global_points": global_points,
                    "global_bbox": (p_xmin, p_ymin, p_xmax, p_ymax),
                    "global_area": float(global_area),
                    "clipped_points_list": clipped_points_list,  # 可能包含多个裁剪片段
                    "clipped_area": float(clipped_area),
                    "area_ratio": float(area_ratio),
                    "is_partial": bool(is_partial),
                    "layer": layer_idx,
                    "patch_bbox": (x_min, y_min, x_max, y_max),
                })

        return geometries
