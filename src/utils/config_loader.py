# src/utils/config_loader.py
import yaml
from pathlib import Path

def load_config(config_file: str) -> dict:
    """加载 YAML 配置文件。

    Args:
        config_file: YAML 配置文件的路径。

    Returns:
        包含配置信息的字典。
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base_config: dict, task_config: dict) -> dict:
    """将特定于任务的配置合并到基础配置中。

    Args:
        base_config: 基础（默认）配置。
        task_config: 要合并的特定于任务的配置。

    Returns:
        合并后的配置字典。
    """
    merged = base_config.copy() # 复制基础配置
    # 遍历任务配置中的键值对
    for key, value in task_config.items():
        # 如果值是字典且键也存在于合并后的配置中，则递归合并
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        # 否则，直接用任务配置的值覆盖
        else:
            merged[key] = value
    return merged
