# src/engine/trainer.py
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import DataLoader
from ..utils.logging import get_logger

class Trainer:
    """处理（监督学习）训练循环。"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # 根据配置选择优化器
        if config['training']['optimizer'] == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        elif config['training']['optimizer'] == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
        else:
            raise ValueError(f"不支持的优化器: {config['training']['optimizer']}")

        # 根据配置选择损失函数
        if config['training']['loss_function'] == 'bce':
            # BCEWithLogitsLoss 结合了 Sigmoid 和 BCELoss，更数值稳定
            self.criterion = nn.BCEWithLogitsLoss()
        # 在此添加其他损失函数，如 focal loss
        else:
            raise ValueError(f"不支持的损失函数: {config['training']['loss_function']}")

    def train_epoch(self, dataloader: DataLoader):
        """运行单个训练周期（epoch）。"""
        self.model.train()  # 将模型设置为训练模式
        total_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()  # 清空梯度
            
            # 前向传播
            output = self.model(batch)
            
            # 准备目标标签
            # 假设标签在图级别，并且需要调整形状以匹配输出
            target = batch.y.view_as(output)

            # 计算损失
            loss = self.criterion(output, target)
            # 反向传播
            loss.backward()
            # 更新权重
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"训练损失: {avg_loss:.4f}")
        return avg_loss

    def run(self, train_loader: DataLoader, val_loader: DataLoader):
        """运行完整的训练流程。"""
        self.logger.info("开始训练...")
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"周期 {epoch+1}/{self.config['training']['epochs']}")
            self.train_epoch(train_loader)
            # 在此处添加验证步骤，例如调用 Evaluator
        self.logger.info("训练完成。")
