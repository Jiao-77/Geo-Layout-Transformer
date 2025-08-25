import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """向输入序列中注入位置信息。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够大的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 创建位置信息 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算用于正弦和余弦函数的分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 计算偶数维度的位置编码（使用正弦）
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算奇数维度的位置编码（使用余弦）
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整形状以匹配输入 [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将 pe 注册为 buffer，这样它不会被视为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 张量，形状为 [seq_len, batch_size, embedding_dim]
        """
        # 将位置编码加到输入张量上
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerCore(nn.Module):
    """用于全局上下文建模的 Transformer 骨干网络。"""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1):
        super(TransformerCore, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        # 定义 Transformer 编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        # 堆叠多个编码器层形成完整的 Transformer 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: 形状为 [batch_size, seq_len, hidden_dim] 的张量，
                              代表所有区块的嵌入向量。

        Returns:
            一个形状为 [batch_size, seq_len, hidden_dim] 的、包含全局上下文信息的张量。
        """
        # 注意：PyTorch 的 TransformerEncoderLayer 期望的输入形状是 (seq_len, batch, features) 
        # 如果 batch_first=False，或者 (batch, seq_len, features) 如果 batch_first=True。
        # 我们的输入是 [batch_size, seq_len, hidden_dim]，所以我们设置 batch_first=True。
        
        # 我们使用的 PositionalEncoding 是为 (seq_len, batch, features) 设计的，所以需要调整一下形状
        src = patch_embeddings.transpose(0, 1) # 转换为 [seq_len, batch_size, hidden_dim]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1) # 转换回 [batch_size, seq_len, hidden_dim]

        # 将带有位置信息的嵌入传入 Transformer
        output = self.transformer_encoder(src)
        return output
