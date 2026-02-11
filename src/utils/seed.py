# src/utils/seed.py
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    设置随机种子，确保实验的可重复性。
    
    Args:
        seed: 随机种子值
    """
    # 设置 Python 内置随机种子
    random.seed(seed)
    
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 对于多 GPU 环境
    
    # 禁用 CUDA 中的确定性算法，以提高性能（可选）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # 设置环境变量中的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子已设置为: {seed}")
