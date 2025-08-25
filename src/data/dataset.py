import torch
from torch_geometric.data import Dataset, InMemoryDataset
import os

class LayoutDataset(InMemoryDataset):
    """用于加载预处理后的版图图数据的 PyTorch Geometric 数据集。"""

    def __init__(self, root, transform=None, pre_transform=None):
        """
        Args:
            root: 数据集应保存的根目录。
            transform: 一个函数/变换，作用于 `Data` 对象并返回一个转换后的版本。
            pre_transform: 一个函数/变换，作用于 `Data` 对象并返回一个转换后的版本。
        """
        super(LayoutDataset, self).__init__(root, transform, pre_transform)
        # 加载已处理的数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """如果 `download()` 返回一个路径列表，这里会返回它们的文件名。"""
        return []  # 我们不从网络下载原始文件

    @property
    def processed_file_names(self):
        """在 `processed_dir` 目录中必须存在的文件列表，用以跳过处理步骤。"""
        return ['data.pt']

    def download(self):
        """从网上下载原始数据到 `raw_dir` 目录。"""
        pass  # 假设数据是预先处理好的

    def process(self):
        """处理原始数据并将其保存到 `processed_dir` 目录。"""
        # 如果希望在加载时动态处理数据，可以在这里实现 `scripts/preprocess_gds.py` 中的逻辑。
        # 在我们的框架中，我们假设预处理是通过脚本独立完成的。
        pass
