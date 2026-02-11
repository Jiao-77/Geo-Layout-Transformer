# src/engine/self_supervised.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..utils.logging import get_logger
import os
import time

class SelfSupervisedTrainer:
    """处理自监督预训练循环（掩码版图建模）。"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.optimizer = AdamW(self.model.parameters(), lr=config['pretraining']['learning_rate'])
        # 使用均方误差损失来重建嵌入向量
        self.criterion = nn.MSELoss()

        # 初始化可学习的 [MASK] 嵌入
        self.mask_embedding = nn.Parameter(torch.randn(config['model']['gnn']['output_dim']))
        # 将其添加到模型参数中，使其可被优化
        self.model.register_parameter('mask_embedding', self.mask_embedding)

        # 初始化重建头
        hidden_dim = config['model']['transformer']['hidden_dim']
        output_dim = config['model']['gnn']['output_dim']
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 确保保存目录存在
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

        # 初始化 TensorBoard 日志记录器
        self.log_dir = config.get('log_dir', 'logs/pretrain')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # 初始化早停相关变量
        self.best_loss = float('inf')
        self.patience = config['pretraining'].get('early_stopping_patience', 10)
        self.counter = 0
        self.early_stop = False
        
        # 初始化混合精度训练
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 初始化梯度累积
        self.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"启用梯度累积，累积步数: {self.gradient_accumulation_steps}")

    def train_epoch(self, dataloader: DataLoader):
        """运行单个预训练周期。"""
        self.model.train()
        self.reconstruction_head.train()
        total_loss = 0
        mask_ratio = self.config['pretraining']['mask_ratio']

        for i, batch in enumerate(dataloader):
            # 只有在梯度累积的第一步或不需要累积时才清空梯度
            if i % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # 使用混合精度训练
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # 1. 获取原始的区块嵌入（作为重建的目标）
                    original_embeddings = self.model.gnn_encoder(batch)

                    # 2. 根据 batch.ptr 逐图生成 mask 索引，避免跨图混淆
                    num_graphs = batch.num_graphs
                    nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
                    
                    # 确保所有图的节点数相同
                    if not torch.all(nodes_per_graph == nodes_per_graph[0]):
                        self.logger.warning("批次中图形的节点数不一致，使用第一个图形的节点数")
                    nodes_per_graph = nodes_per_graph[0]
                    
                    # 为每个图单独生成掩码
                    all_masked_indices = []
                    for j in range(num_graphs):
                        # 计算当前图的节点在批次中的起始和结束索引
                        start_idx = batch.ptr[j]
                        end_idx = batch.ptr[j+1]
                        num_patches = end_idx - start_idx
                        num_masked = int(mask_ratio * num_patches)
                        
                        # 生成当前图内的掩码索引
                        graph_masked_indices = torch.randperm(num_patches)[:num_masked] + start_idx
                        all_masked_indices.append(graph_masked_indices)
                    
                    # 合并所有图的掩码索引
                    masked_indices = torch.cat(all_masked_indices)

                    # 3. 创建损坏的嵌入
                    corrupted_embeddings = original_embeddings.clone()
                    # 使用可学习的 [MASK] 嵌入
                    corrupted_embeddings[masked_indices] = self.mask_embedding.to(corrupted_embeddings.device)

                    # 4. 为 Transformer 重塑形状
                    corrupted_embeddings = corrupted_embeddings.view(num_graphs, nodes_per_graph, -1)

                    # 5. 将损坏的嵌入传入 Transformer 进行编码
                    encoded_embeddings = self.model.transformer_core(corrupted_embeddings)

                    # 6. 通过重建头生成重建的嵌入
                    reconstructed_embeddings = self.reconstruction_head(encoded_embeddings)

                    # 7. 只在被掩盖的区块上计算损失
                    # 将 Transformer 输出和原始嵌入都拉平成 (N, D) 的形状
                    reconstructed_flat = reconstructed_embeddings.view(-1, original_embeddings.size(1))
                    # 只选择被掩盖的那些进行比较
                    loss = self.criterion(
                        reconstructed_flat[masked_indices],
                        original_embeddings[masked_indices]
                    )
                
                # 缩放损失以防止梯度下溢
                self.scaler.scale(loss).backward()
                
                # 只有在累积步数达到设定值时才更新权重
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # 取消缩放并更新权重
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # 标准训练流程
                # 1. 获取原始的区块嵌入（作为重建的目标）
                original_embeddings = self.model.gnn_encoder(batch)

                # 2. 根据 batch.ptr 逐图生成 mask 索引，避免跨图混淆
                num_graphs = batch.num_graphs
                nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
                
                # 确保所有图的节点数相同
                if not torch.all(nodes_per_graph == nodes_per_graph[0]):
                    self.logger.warning("批次中图形的节点数不一致，使用第一个图形的节点数")
                nodes_per_graph = nodes_per_graph[0]
                
                # 为每个图单独生成掩码
                all_masked_indices = []
                for j in range(num_graphs):
                    # 计算当前图的节点在批次中的起始和结束索引
                    start_idx = batch.ptr[j]
                    end_idx = batch.ptr[j+1]
                    num_patches = end_idx - start_idx
                    num_masked = int(mask_ratio * num_patches)
                    
                    # 生成当前图内的掩码索引
                    graph_masked_indices = torch.randperm(num_patches)[:num_masked] + start_idx
                    all_masked_indices.append(graph_masked_indices)
                
                # 合并所有图的掩码索引
                masked_indices = torch.cat(all_masked_indices)

                # 3. 创建损坏的嵌入
                corrupted_embeddings = original_embeddings.clone()
                # 使用可学习的 [MASK] 嵌入
                corrupted_embeddings[masked_indices] = self.mask_embedding.to(corrupted_embeddings.device)

                # 4. 为 Transformer 重塑形状
                corrupted_embeddings = corrupted_embeddings.view(num_graphs, nodes_per_graph, -1)

                # 5. 将损坏的嵌入传入 Transformer 进行编码
                encoded_embeddings = self.model.transformer_core(corrupted_embeddings)

                # 6. 通过重建头生成重建的嵌入
                reconstructed_embeddings = self.reconstruction_head(encoded_embeddings)

                # 7. 只在被掩盖的区块上计算损失
                # 将 Transformer 输出和原始嵌入都拉平成 (N, D) 的形状
                reconstructed_flat = reconstructed_embeddings.view(-1, original_embeddings.size(1))
                # 只选择被掩盖的那些进行比较
                loss = self.criterion(
                    reconstructed_flat[masked_indices],
                    original_embeddings[masked_indices]
                )

                loss.backward()
                
                # 只有在累积步数达到设定值时才更新权重
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    # 更新权重
                    self.optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"预训练损失: {avg_loss:.4f}")
        return avg_loss

    def run(self, train_loader: DataLoader):
        """运行完整的预训练流程。"""
        self.logger.info("开始自监督预训练...")
        start_time = time.time()
        
        for epoch in range(self.config['pretraining']['epochs']):
            if self.early_stop:
                self.logger.info("早停触发，停止预训练。")
                break
                
            epoch_start_time = time.time()
            self.logger.info(f"周期 {epoch+1}/{self.config['pretraining']['epochs']}")
            current_loss = self.train_epoch(train_loader)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录到 TensorBoard
            self.writer.add_scalar('Loss/pretrain', current_loss, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # 计算周期耗时
            epoch_time = time.time() - epoch_start_time
            self.writer.add_scalar('Time/epoch', epoch_time, epoch)
            self.logger.info(f"周期耗时: {epoch_time:.2f} 秒")
            
            # 检查是否需要保存最佳模型
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.counter = 0
                # 保存最佳模型
                save_path = os.path.join(self.save_dir, 'best_pretrain_model.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'reconstruction_head_state_dict': self.reconstruction_head.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss
                }, save_path)
                self.logger.info(f"保存最佳预训练模型到 {save_path}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.logger.info(f"预训练损失连续 {self.patience} 个周期未改善，触发早停。")
        
        # 计算总训练耗时
        total_time = time.time() - start_time
        self.logger.info(f"总预训练耗时: {total_time:.2f} 秒")
        
        # 保存最后一个模型
        save_path = os.path.join(self.save_dir, 'last_pretrain_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'reconstruction_head_state_dict': self.reconstruction_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
        self.logger.info(f"保存最后一个预训练模型到 {save_path}")
        
        # 关闭 TensorBoard SummaryWriter
        self.writer.close()
        
        self.logger.info("预训练完成。")
        self.logger.info(f"最佳预训练损失: {self.best_loss:.4f}")
