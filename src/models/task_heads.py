import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """一个用于分类任务的简单多层感知机（MLP）任务头。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ClassificationHead, self).__init__()
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
        # 我们可以取第一个 token（类似 [CLS]）的嵌入，或者进行平均池化
        # 为简单起见，我们假设在序列维度上进行平均池化
        x_pooled = torch.mean(x, dim=1)
        
        out = self.fc1(x_pooled)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MatchingHead(nn.Module):
    """用于学习版图匹配的相似性嵌入的任务头。"""

    def __init__(self, input_dim: int, output_dim: int):
        super(MatchingHead, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 来自 Transformer 骨干网络的输入张量。

        Returns:
            代表整个输入图（例如一个 IP 模块）的单个嵌入向量。
        """
        # 全局平均池化，为整个序列获取一个单一的向量
        graph_embedding = torch.mean(x, dim=1)
        # 投影到最终的嵌入空间
        similarity_embedding = self.projection(graph_embedding)
        # 对嵌入进行 L2 归一化，以便使用余弦相似度
        similarity_embedding = nn.functional.normalize(similarity_embedding, p=2, dim=1)
        return similarity_embedding
