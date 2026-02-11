#!/usr/bin/env python3
"""
一键运行的小样流程脚本
- 生成示例数据
- 训练模型
- 评估模型
"""
import os
import sys
import subprocess
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd, cwd=None):
    """运行命令并打印输出"""
    print(f"\n运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print("输出:")
    print(result.stdout)
    if result.stderr:
        print("错误:")
        print(result.stderr)
    if result.returncode != 0:
        print(f"命令执行失败，返回码: {result.returncode}")
        sys.exit(1)
    return result

def generate_sample_data():
    """生成示例数据"""
    print("\n=== 步骤 1: 生成示例数据 ===")
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_sample_data.py")
    run_command([sys.executable, script_path])
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")

def train_model(data_dir):
    """训练模型"""
    print("\n=== 步骤 2: 训练模型 ===")
    main_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "hotspot_detection.yaml")
    
    # 运行训练命令
    cmd = [
        sys.executable, main_script,
        "--config-file", config_file,
        "--mode", "train",
        "--data-dir", data_dir
    ]
    run_command(cmd)

def evaluate_model(data_dir):
    """评估模型"""
    print("\n=== 步骤 3: 评估模型 ===")
    main_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "hotspot_detection.yaml")
    
    # 运行评估命令
    cmd = [
        sys.executable, main_script,
        "--config-file", config_file,
        "--mode", "eval",
        "--data-dir", data_dir
    ]
    run_command(cmd)

def main():
    """主函数"""
    start_time = time.time()
    
    print("Geo-Layout Transformer 小样流程")
    print("==============================")
    
    # 步骤 1: 生成示例数据
    data_dir = generate_sample_data()
    
    # 步骤 2: 训练模型
    train_model(data_dir)
    
    # 步骤 3: 评估模型
    evaluate_model(data_dir)
    
    total_time = time.time() - start_time
    print(f"\n=== 流程完成 ===")
    print(f"总耗时: {total_time:.2f} 秒")
    print("示例流程已成功运行!")

if __name__ == "__main__":
    main()
