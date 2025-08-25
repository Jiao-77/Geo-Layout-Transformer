import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import DataLoader
from ..utils.logging import get_logger

class SelfSupervisedTrainer:
    """处理自监督预训练循环（掩码版图建模）。"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.optimizer = AdamW(self.model.parameters(), lr=config['pretraining']['learning_rate'])
        # 使用均方误差损失来重建嵌入向量
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader):
        """运行单个预训练周期。"""
        self.model.train()
        total_loss = 0
        mask_ratio = self.config['pretraining']['mask_ratio']

        for batch in dataloader:
            self.optimizer.zero_grad()

            # 1. 获取原始的区块嵌入（作为重建的目标）
            with torch.no_grad():
                original_embeddings = self.model.gnn_encoder(batch)

            # 2. 创建掩码并损坏输入
            num_patches = original_embeddings.size(0)
            num_masked = int(mask_ratio * num_patches)
            # 随机选择要掩盖的区块索引
            masked_indices = torch.randperm(num_patches)[:num_masked]
            
            # 创建一个损坏的嵌入副本
            # 这是一个简化的方法。更稳健的方法是直接在批次数据中掩盖特征。
            # 在这个占位符中，我们直接掩盖嵌入向量。
            corrupted_embeddings = original_embeddings.clone()
            # 创建一个可学习的 [MASK] 嵌入
            mask_embedding = nn.Parameter(torch.randn(original_embeddings.size(1), device=original_embeddings.device))
            corrupted_embeddings[masked_indices] = mask_embedding

            # 3. 为 Transformer 重塑形状
            num_graphs = batch.num_graphs
            nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
            corrupted_embeddings = corrupted_embeddings.view(num_graphs, nodes_per_graph[0], -1)

            # 4. 将损坏的嵌入传入 Transformer 进行重建
            # 注意：这里只用了 transformer_core，没有用 task_head
            reconstructed_embeddings = self.model.transformer_core(corrupted_embeddings)

            # 5. 只在被掩盖的区块上计算损失
            # 将 Transformer 输出和原始嵌入都拉平成 (N, D) 的形状
            reconstructed_flat = reconstructed_embeddings.view(-1, original_embeddings.size(1))
            # 只选择被掩盖的那些进行比较
            loss = self.criterion(
                reconstructed_flat[masked_indices],
                original_embeddings[masked_indices]
            )

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"预训练损失: {avg_loss:.4f}")
        return avg_loss

    def run(self, train_loader: DataLoader):
        """运行完整的预训练流程。"""
        self.logger.info("开始自监督预训练...")
        for epoch in range(self.config['pretraining']['epochs']):
            self.logger.info(f"周期 {epoch+1}/{self.config['pretraining']['epochs']}")
            self.train_epoch(train_loader)
        self.logger.info("预训练完成。")
