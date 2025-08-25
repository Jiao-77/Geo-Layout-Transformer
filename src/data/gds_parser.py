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
        """从给定的区块中提取所有几何对象。

        Args:
            patch_bbox: 区块的边界框 (x_min, y_min, x_max, y_max)。

        Returns:
            一个字典列表，每个字典代表一个几何对象及其属性（多边形、层、边界框）。
        """
        x_min, y_min, x_max, y_max = patch_bbox
        # 获取单元内的所有多边形
        polygons = self.top_cell.get_polygons(by_spec=True)
        geometries = []
        # 遍历所有多边形
        for (layer, datatype), poly_list in polygons.items():
            layer_str = f"{layer}/{datatype}"
            # 只处理在 layer_mapping 中定义的层
            if layer_str in self.layer_mapping:
                for poly in poly_list:
                    # 简单的边界框相交检查
                    p_xmin, p_ymin, p_xmax, p_ymax = poly.bb()
                    if not (p_xmax < x_min or p_xmin > x_max or p_ymax < y_min or p_ymin > y_max):
                        geometries.append({
                            "polygon": poly,
                            "layer": self.layer_mapping[layer_str],
                            "bbox": (p_xmin, p_ymin, p_xmax, p_ymax)
                        })
        return geometries
