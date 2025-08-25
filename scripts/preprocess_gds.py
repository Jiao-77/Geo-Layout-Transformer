import argparse
import os
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data

from src.utils.config_loader import load_config
from src.data.gds_parser import GDSParser
from src.data.graph_constructor import GraphConstructor
from src.utils.logging import get_logger

# 这是一个辅助的数据集类，仅用于在预处理脚本中保存数据
class TempDataset(InMemoryDataset):
    def __init__(self, root, data_list=None):
        self.data_list = data_list
        super(TempDataset, self).__init__(root)
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # 数据已在外部处理好，直接保存
        torch.save((self.data, self.slices), self.processed_paths[0])

def main():
    parser = argparse.ArgumentParser(description="将 GDSII 文件预处理为图数据。")
    parser.add_argument("--config-file", required=True, help="配置文件的路径。")
    parser.add_argument("--gds-file", required=True, help="要处理的 GDSII 文件的路径。")
    parser.add_argument("--output-dir", required=True, help="保存处理后图数据的目录。")
    # 可以添加一个参数来指定标签文件，例如 DRC 报告
    # parser.add_argument("--label-file", help="标签文件的路径。")
    args = parser.parse_args()

    logger = get_logger("GDS_Preprocessor")

    logger.info(f"从 {args.config_file} 加载配置")
    config = load_config(args.config_file)

    logger.info(f"为 {args.gds_file} 初始化 GDSParser")
    gds_parser = GDSParser(args.gds_file, config['data']['layer_mapping'])

    logger.info("初始化 GraphConstructor")
    graph_constructor = GraphConstructor(
        edge_strategy=config['data']['graph_construction']['edge_strategy'],
        knn_k=config['data']['graph_construction']['knn_k'],
        radius_d=config['data']['graph_construction']['radius_d']
    )

    logger.info("正在生成区块...")
    patches = gds_parser.get_patches(config['data']['patch_size'], config['data']['patch_stride'])
    logger.info(f"生成了 {len(patches)} 个区块。")

    os.makedirs(args.output_dir, exist_ok=True)

    graph_list = []
    # 使用 tqdm 显示进度条
    for patch_bbox in tqdm(patches, desc="处理区块中"):
        geometries = gds_parser.extract_geometries_from_patch(patch_bbox)
        if geometries:
            # 在真实场景中，您需要从 DRC 报告等来源获取标签
            # 在这个占位符中，我们假设一个虚拟标签 0
            # TODO: 实现从标签文件加载标签的逻辑
            graph = graph_constructor.construct_graph(geometries, label=0)
            if graph:
                # PyG 要求 Data 对象具有 y 属性
                if not hasattr(graph, 'y'):
                    graph.y = torch.tensor([0], dtype=torch.float)
                graph_list.append(graph)

    logger.info(f"成功构建了 {len(graph_list)} 个图。")

    if graph_list:
        # 使用 PyG 的 InMemoryDataset 格式保存数据，以便高效加载
        logger.info("正在将数据保存为 PyG InMemoryDataset 格式...")
        dataset = TempDataset(root=args.output_dir, data_list=graph_list)
        logger.info(f"已将处理好的数据保存到 {dataset.processed_paths[0]}")
    else:
        logger.warning("没有生成任何图数据，不进行保存。")

if __name__ == "__main__":
    main()
