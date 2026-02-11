# scripts/visualize_attention.py
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils.config_loader import load_config
from src.models.geo_layout_transformer import GeoLayoutTransformer
from src.utils.logging import get_logger

def main():
    parser = argparse.ArgumentParser(description="可视化来自已训练模型的注意力图。")
    parser.add_argument("--config-file", required=True, help="模型配置文件的路径。")
    parser.add_argument("--model-path", required=True, help="已训练模型检查点的路径。")
    parser.add_argument("--patch-data", required=True, help="区块数据样本（.pt 文件）的路径。")
    parser.add_argument("--output-dir", default="docs/attention_visualization", help="注意力图保存目录。")
    parser.add_argument("--layer-index", type=int, default=0, help="要可视化的 Transformer 层索引。")
    parser.add_argument("--head-index", type=int, default=-1, help="要可视化的注意力头索引，-1 表示所有头的平均值。")
    args = parser.parse_args()

    logger = get_logger("Attention_Visualizer")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载配置和模型
    logger.info("正在加载模型...")
    config = load_config(args.config_file)
    model = GeoLayoutTransformer(config)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. 加载一个数据样本
    logger.info(f"正在加载数据样本从 {args.patch_data}")
    sample_data = torch.load(args.patch_data)

    # 3. 注册钩子（Hook）到模型中以提取注意力权重
    attention_weights = []
    
    def hook(module, input, output):
        # 对于 PyTorch 的 nn.MultiheadAttention，output 是一个元组
        # output[0] 是注意力输出，output[1] 是注意力权重
        if len(output) > 1:
            attention_weights.append(output[1])
    
    # 获取指定层的自注意力模块
    if hasattr(model.transformer_core.transformer_encoder, 'layers'):
        layer = model.transformer_core.transformer_encoder.layers[args.layer_index]
        if hasattr(layer, 'self_attn'):
            layer.self_attn.register_forward_hook(hook)
            logger.info(f"已注册钩子到 Transformer 层 {args.layer_index} 的自注意力模块")
        else:
            logger.error("找不到自注意力模块")
            return
    else:
        logger.error("找不到 Transformer 层")
        return

    # 4. 运行一次前向传播以获取权重
    logger.info("正在运行前向传播...")
    with torch.no_grad():
        _ = model(sample_data)

    # 5. 绘制注意力图
    if attention_weights:
        logger.info("正在绘制注意力图...")
        # attention_weights[0] 的形状是 [batch_size, num_heads, seq_len, seq_len]
        attn_weights = attention_weights[0]
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        logger.info(f"注意力权重形状: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}")
        
        # 选择第一个样本
        sample_attn = attn_weights[0]
        
        if args.head_index == -1:
            # 计算所有头的平均值
            avg_attention = sample_attn.mean(dim=0).cpu().numpy()
            plt.figure(figsize=(12, 10))
            sns.heatmap(avg_attention, cmap='viridis', square=True, vmin=0, vmax=1)
            plt.title(f"所有注意力头的平均注意力图 (Layer {args.layer_index})")
            plt.xlabel("区块索引")
            plt.ylabel("区块索引")
            output_file = os.path.join(args.output_dir, f"attention_layer_{args.layer_index}_avg.png")
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            logger.info(f"已保存平均注意力图到 {output_file}")
        else:
            # 可视化指定的注意力头
            if 0 <= args.head_index < num_heads:
                head_attention = sample_attn[args.head_index].cpu().numpy()
                plt.figure(figsize=(12, 10))
                sns.heatmap(head_attention, cmap='viridis', square=True, vmin=0, vmax=1)
                plt.title(f"注意力头 {args.head_index} 的注意力图 (Layer {args.layer_index})")
                plt.xlabel("区块索引")
                plt.ylabel("区块索引")
                output_file = os.path.join(args.output_dir, f"attention_layer_{args.layer_index}_head_{args.head_index}.png")
                plt.savefig(output_file, bbox_inches='tight', dpi=150)
                logger.info(f"已保存注意力头 {args.head_index} 的注意力图到 {output_file}")
            else:
                logger.error(f"注意力头索引 {args.head_index} 超出范围，有效范围是 0-{num_heads-1}")
    else:
        logger.warning("未能提取注意力权重。")

if __name__ == "__main__":
    main()
