# src/engine/trainer.py
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..utils.logging import get_logger
from .evaluator import Evaluator
import os
import time

class FocalLoss(nn.Module):
    """Focal Loss 实现，用于处理类别不平衡问题。"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
        elif config['training']['loss_function'] == 'focal_loss':
            self.criterion = FocalLoss()
        else:
            raise ValueError(f"不支持的损失函数: {config['training']['loss_function']}")

        # 初始化学习率调度器
        self.scheduler = None
        if 'scheduler' in config['training']:
            scheduler_type = config['training']['scheduler']
            if scheduler_type == 'step':
                self.scheduler = StepLR(self.optimizer, step_size=config['training'].get('scheduler_step_size', 30), gamma=config['training'].get('scheduler_gamma', 0.1))
            elif scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=config['training'].get('scheduler_T_0', 10), T_mult=config['training'].get('scheduler_T_mult', 2))

        # 初始化评估器
        self.evaluator = Evaluator(model)

        # 初始化早停相关变量
        self.best_val_score = -float('inf')
        self.patience = config['training'].get('early_stopping_patience', 10)
        self.counter = 0
        self.early_stop = False

        # 确保保存目录存在
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化 TensorBoard 日志记录器
        self.log_dir = config.get('log_dir', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 初始化混合精度训练
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 初始化梯度累积
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"启用梯度累积，累积步数: {self.gradient_accumulation_steps}")

    def train_epoch(self, dataloader: DataLoader):
        """运行单个训练周期（epoch）。"""
        self.model.train()  # 将模型设置为训练模式
        total_loss = 0
        
        for i, batch in enumerate(dataloader):
            # 只有在梯度累积的第一步或不需要累积时才清空梯度
            if i % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # 使用混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # 前向传播
                    output = self.model(batch)
                    
                    # 准备目标标签
                    # 假设标签在图级别，并且需要调整形状以匹配输出
                    target = batch.y.view_as(output)

                    # 计算损失
                    loss = self.criterion(output, target)
                
                # 缩放损失以防止梯度下溢
                self.scaler.scale(loss).backward()
                
                # 只有在累积步数达到设定值时才更新权重
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # 取消缩放并更新权重
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # 标准训练流程
                # 前向传播
                output = self.model(batch)
                
                # 准备目标标签
                # 假设标签在图级别，并且需要调整形状以匹配输出
                target = batch.y.view_as(output)

                # 计算损失
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 只有在累积步数达到设定值时才更新权重
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # 更新权重
                    self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"训练损失: {avg_loss:.4f}")
        return avg_loss

    def validate(self, dataloader: DataLoader):
        """运行验证并返回评估指标。"""
        self.model.eval()  # 将模型设置为评估模式
        metrics = self.evaluator.evaluate(dataloader)
        return metrics

    def run(self, train_loader: DataLoader, val_loader: DataLoader):
        """运行完整的训练流程。"""
        self.logger.info("开始训练...")
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            if self.early_stop:
                self.logger.info("早停触发，停止训练。")
                break
                
            epoch_start_time = time.time()
            self.logger.info(f"周期 {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练一个周期
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            self.logger.info("正在验证...")
            val_metrics = self.validate(val_loader)
            
            # 更新学习率调度器
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f"学习率从 {current_lr:.6f} 调整为 {new_lr:.6f}")
                current_lr = new_lr
            else:
                self.logger.info(f"当前学习率: {current_lr:.6f}")
            
            # 记录到 TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # 计算周期耗时
            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Time/epoch', epoch_time, epoch)
            self.logger.info(f"周期耗时: {epoch_time:.2f} 秒")
            
            # 检查是否需要保存最佳模型
            val_score = val_metrics.get('f1', val_metrics.get('accuracy', -1))
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.counter = 0
                # 保存最佳模型
                save_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f"保存最佳模型到 {save_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.logger.info(f"验证性能连续 {self.patience} 个周期未改善，触发早停。")
        
        # 计算总训练耗时
        total_time = time.time() - start_time
        self.logger.info(f"总训练耗时: {total_time:.2f} 秒")
        
        # 保存最后一个模型
        save_path = os.path.join(self.save_dir, 'last_model.pth')
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"保存最后一个模型到 {save_path}")
        
        # 关闭 TensorBoard SummaryWriter
        self.writer.close()
        
        self.logger.info("训练完成。")
        self.logger.info(f"最佳验证分数: {self.best_val_score:.4f}")
