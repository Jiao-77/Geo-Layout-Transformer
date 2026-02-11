# main.py
import argparse
from torch.utils.data import random_split

from src.utils.config_loader import load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.data.dataset import LayoutDataset
from torch_geometric.data import DataLoader
from src.models.geo_layout_transformer import GeoLayoutTransformer
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.engine.self_supervised import SelfSupervisedTrainer

def main():
    parser = argparse.ArgumentParser(description="Geo-Layout Transformer 的主脚本。")
    parser.add_argument("--config-file", required=True, help="特定于任务的配置文件的路径。")
    parser.add_argument("--mode", choices=["train", "eval", "pretrain"], required=True, help="脚本运行模式。")
    parser.add_argument("--data-dir", required=True, help="已处理图数据的目录。")
    parser.add_argument("--checkpoint-path", help="要加载的模型检查点的路径。")
    args = parser.parse_args()

    logger = get_logger("Main")

    # 加载配置
    logger.info("正在加载配置...")
    # 首先加载基础配置，然后用任务特定配置覆盖
    base_config = load_config('configs/default.yaml')
    task_config = load_config(args.config_file)
    config = merge_configs(base_config, task_config)
    
    # 设置随机种子，确保实验的可重复性
    random_seed = config['splits']['random_seed']
    logger.info(f"正在设置随机种子: {random_seed}")
    set_seed(random_seed)

    # 加载数据
    logger.info(f"从 {args.data_dir} 加载数据集")
    dataset = LayoutDataset(root=args.data_dir)
    
    # 实现数据集划分逻辑
    logger.info("正在划分数据集...")
    train_ratio = config['splits']['train_ratio']
    val_ratio = config['splits']['val_ratio']
    test_ratio = config['splits']['test_ratio']
    random_seed = config['splits']['random_seed']
    
    # 计算各数据集大小
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # 确保各部分大小合理
    if test_size < 0:
        test_size = 0
        val_size = len(dataset) - train_size
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    logger.info(f"数据集划分完成: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    # 初始化模型
    logger.info("正在初始化模型...")
    model = GeoLayoutTransformer(config)
    if args.checkpoint_path:
        logger.info(f"从 {args.checkpoint_path} 加载模型检查点")
        # model.load_state_dict(torch.load(args.checkpoint_path))

    # 根据模式运行
    if args.mode == 'pretrain':
        logger.info("进入自监督预训练模式...")
        trainer = SelfSupervisedTrainer(model, config)
        trainer.run(train_loader)
    elif args.mode == 'train':
        logger.info("进入监督训练模式...")
        trainer = Trainer(model, config)
        trainer.run(train_loader, val_loader)
    elif args.mode == 'eval':
        logger.info("进入评估模式...")
        evaluator = Evaluator(model)
        evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()
