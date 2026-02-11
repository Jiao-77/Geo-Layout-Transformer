# src/models/task_heads.py
import torch
import torch.nn as nn

class PoolingLayer(nn.Module):
    """可插拔的池化层，支持多种池化策略。"""
    def __init__(self, pooling_type: str = 'mean'):
        super(PoolingLayer, self).__init__()
        self.pooling_type = pooling_type
        
        # 如果使用注意力池化，需要定义注意力机制
        if pooling_type == 'attention':
            self.attention = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状为 [batch_size, seq_len, hidden_dim] 的张量

        Returns:
            形状为 [batch_size, hidden_dim] 的池化后的张量
        """
        if self.pooling_type == 'mean':
            return torch.mean(x, dim=1)
        elif self.pooling_type == 'max':
            return torch.max(x, dim=1)[0]
        elif self.pooling_type == 'cls':
            # 取第一个 token 作为 [CLS] token
            return x[:, 0, :]
        elif self.pooling_type == 'attention':
            # 计算注意力权重
            weights = self.attention(torch.ones_like(x[:, :, :1])).softmax(dim=1)
            return (x * weights).sum(dim=1)
        else:
            raise ValueError(f"不支持的池化类型: {self.pooling_type}")

class ClassificationHead(nn.Module):
    """一个用于分类任务的简单多层感知机（MLP）任务头。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, pooling_type: str = 'mean'):
        super(ClassificationHead, self).__init__()
        self.pooling = PoolingLayer(pooling_type)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 来自 Transformer 骨干网络的输入张量。

        Returns:
            最终的分类 logits。
        """
        # 使用指定的池化方法
        x_pooled = self.pooling(x)
        
        out = self.fc1(x_pooled)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MultiLabelClassificationHead(nn.Module):
    """用于多标签分类任务的任务头。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, pooling_type: str = 'mean'):
        super(MultiLabelClassificationHead, self).__init__()
        self.pooling = PoolingLayer(pooling_type)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 来自 Transformer 骨干网络的输入张量。

        Returns:
            最终的多标签分类 logits。
        """
        # 使用指定的池化方法
        x_pooled = self.pooling(x)
        
        out = self.fc1(x_pooled)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class RegressionHead(nn.Module):
    """用于回归任务的任务头。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, pooling_type: str = 'mean'):
        super(RegressionHead, self).__init__()
        self.pooling = PoolingLayer(pooling_type)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 来自 Transformer 骨干网络的输入张量。

        Returns:
            最终的回归输出。
        """
        # 使用指定的池化方法
        x_pooled = self.pooling(x)
        
        out = self.fc1(x_pooled)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MatchingHead(nn.Module):
    """用于学习版图匹配的相似性嵌入的任务头。"""

    def __init__(self, input_dim: int, output_dim: int, pooling_type: str = 'mean'):
        super(MatchingHead, self).__init__()
        self.pooling = PoolingLayer(pooling_type)
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 来自 Transformer 骨干网络的输入张量。

        Returns:
            代表整个输入图（例如一个 IP 模块）的单个嵌入向量。
        """
        # 使用指定的池化方法
        graph_embedding = self.pooling(x)
        # 投影到最终的嵌入空间
        similarity_embedding = self.projection(graph_embedding)
        # 对嵌入进行 L2 归一化，以便使用余弦相似度
        similarity_embedding = nn.functional.normalize(similarity_embedding, p=2, dim=1)
        return similarity_embedding
