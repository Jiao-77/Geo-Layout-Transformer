#!/usr/bin/env python3
"""
测试脚本，用于验证模型是否可以正常跑通，不需要真实数据
- 生成随机图数据
- 加载模型配置
- 初始化模型
- 运行前向传播和反向传播
- 验证模型是否可以正常工作
"""
import os
import sys
import torch
from torch_geometric.data import Data, Batch

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config
from src.models.geo_layout_transformer import GeoLayoutTransformer
from src.engine.trainer import Trainer
from src.engine.self_supervised import SelfSupervisedTrainer
from src.utils.logging import get_logger

def generate_random_graph_data(num_graphs=4, num_nodes_per_graph=8, node_feature_dim=5, edge_feature_dim=0):
    """
    生成随机的图数据
    
    Args:
        num_graphs: 图的数量
        num_nodes_per_graph: 每个图的节点数量
        node_feature_dim: 节点特征维度
        edge_feature_dim: 边特征维度
    
    Returns:
        一个 Batch 对象，包含多个随机生成的图
    """
    graphs = []
    
    for _ in range(num_graphs):
        # 生成随机节点特征
        x = torch.randn(num_nodes_per_graph, node_feature_dim)
        
        # 生成随机边（完全连接）
        edge_index = []
        for i in range(num_nodes_per_graph):
            for j in range(num_nodes_per_graph):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # 生成随机标签
        y = torch.randn(1, 1)  # 假设是图级别的标签
        
        # 创建图数据
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)
    
    # 构建批次
    batch = Batch.from_data_list(graphs)
    return batch

def test_supervised_training():
    """测试监督训练"""
    logger = get_logger("Test_Supervised_Training")
    logger.info("=== 测试监督训练 ===")
    
    # 加载配置
    config = load_config('configs/default.yaml')
    
    # 生成随机数据
    batch = generate_random_graph_data()
    logger.info(f"生成的批次数据: {batch}")
    logger.info(f"批次大小: {batch.num_graphs}")
    logger.info(f"总节点数: {batch.num_nodes}")
    logger.info(f"总边数: {batch.num_edges}")
    
    # 初始化模型
    logger.info("初始化模型...")
    model = GeoLayoutTransformer(config)
    logger.info("模型初始化成功")
    
    # 初始化训练器
    logger.info("初始化训练器...")
    trainer = Trainer(model, config)
    logger.info("训练器初始化成功")
    
    # 测试前向传播
    logger.info("测试前向传播...")
    with torch.no_grad():
        # 先测试 GNN 编码器
        gnn_output = model.gnn_encoder(batch)
        logger.info(f"GNN 编码器输出形状: {gnn_output.shape}")
        
        # 测试形状重塑
        num_graphs = batch.num_graphs
        nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        logger.info(f"每个图的节点数: {nodes_per_graph}")
        reshaped_embeddings = gnn_output.view(num_graphs, nodes_per_graph[0], -1)
        logger.info(f"重塑后的嵌入形状: {reshaped_embeddings.shape}")
        
        # 测试 Transformer 核心
        transformer_output = model.transformer_core(reshaped_embeddings)
        logger.info(f"Transformer 输出形状: {transformer_output.shape}")
        
        # 测试完整模型
        output = model(batch)
    logger.info(f"前向传播成功，输出形状: {output.shape}")
    
    # 测试反向传播
    logger.info("测试反向传播...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.zero_grad()
    output = model(batch)
    
    # 对输出进行全局池化，得到图级别的表示
    # 从 [batch_size, seq_len, hidden_dim] 变为 [batch_size, hidden_dim]
    graph_output = output.mean(dim=1)
    
    # 使用 MSE 损失，只比较前 1 个维度（与 batch.y 形状匹配）
    loss = torch.nn.MSELoss()(graph_output[:, :1], batch.y)
    loss.backward()
    optimizer.step()
    logger.info(f"反向传播成功，损失值: {loss.item()}")
    
    logger.info("监督训练测试完成，模型可以正常工作！")

def test_self_supervised_training():
    """测试自监督训练"""
    logger = get_logger("Test_Self_Supervised_Training")
    logger.info("\n=== 测试自监督训练 ===")
    
    # 加载配置
    config = load_config('configs/default.yaml')
    
    # 生成随机数据
    batch = generate_random_graph_data()
    logger.info(f"生成的批次数据: {batch}")
    logger.info(f"批次大小: {batch.num_graphs}")
    logger.info(f"总节点数: {batch.num_nodes}")
    logger.info(f"总边数: {batch.num_edges}")
    
    # 初始化模型
    logger.info("初始化模型...")
    model = GeoLayoutTransformer(config)
    logger.info("模型初始化成功")
    
    # 初始化自监督训练器
    logger.info("初始化自监督训练器...")
    trainer = SelfSupervisedTrainer(model, config)
    logger.info("自监督训练器初始化成功")
    
    # 测试前向传播
    logger.info("测试前向传播...")
    with torch.no_grad():
        # 测试 GNN 编码器
        gnn_output = model.gnn_encoder(batch)
        logger.info(f"GNN 编码器输出形状: {gnn_output.shape}")
        
        # 测试 Transformer 核心
        num_graphs = batch.num_graphs
        nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        if not torch.all(nodes_per_graph == nodes_per_graph[0]):
            logger.warning("批次中图形的节点数不一致，使用第一个图形的节点数")
        nodes_per_graph = nodes_per_graph[0]
        
        gnn_output_reshaped = gnn_output.view(num_graphs, nodes_per_graph, -1)
        transformer_output = model.transformer_core(gnn_output_reshaped)
        logger.info(f"Transformer 核心输出形状: {transformer_output.shape}")
    
    # 测试完整模型前向传播
    logger.info("测试完整模型前向传播...")
    with torch.no_grad():
        output = model(batch)
    logger.info(f"完整模型前向传播成功，输出形状: {output.shape}")
    
    logger.info("自监督训练测试完成，模型可以正常工作！")

def main():
    """主函数"""
    logger = get_logger("Test_Model_Run")
    logger.info("开始测试模型是否可以正常跑通...")
    
    try:
        # 测试监督训练
        test_supervised_training()
        
        # 测试自监督训练
        test_self_supervised_training()
        
        logger.info("\n✅ 所有测试通过，模型可以正常跑通！")
        logger.info("模型已准备就绪，可以使用真实数据进行训练。")
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
