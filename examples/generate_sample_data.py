#!/usr/bin/env python3
"""
生成示例数据的脚本
- 创建一个简单的 GDS 文件
- 使用 preprocess_gds.py 处理它，生成示例数据集
"""
import os
import sys
import gdstk
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_simple_gds(output_file):
    """创建一个简单的 GDS 文件，包含几个矩形"""
    # 创建一个新的库
    lib = gdstk.Library("simple_layout")
    
    # 创建一个新的单元
    top_cell = lib.new_cell("TOP")
    
    # 在不同层上添加几个矩形
    # 层 1: 金属层 1
    rect1 = gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0)
    top_cell.add(rect1)
    
    # 层 2: 过孔层
    via = gdstk.rectangle((4, 4), (6, 6), layer=2, datatype=0)
    top_cell.add(via)
    
    # 层 3: 金属层 2
    rect2 = gdstk.rectangle((2, 2), (8, 8), layer=3, datatype=0)
    top_cell.add(rect2)
    
    # 保存 GDS 文件
    lib.write_gds(output_file)
    print(f"已创建 GDS 文件: {output_file}")

def preprocess_sample_data(gds_file, output_dir):
    """使用 preprocess_gds.py 处理 GDS 文件，生成示例数据集"""
    import subprocess
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行 preprocess_gds.py 脚本
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "preprocess_gds.py")
    
    # 创建层映射配置
    layer_mapping = {
        "1/0": 0,  # 金属层 1
        "2/0": 1,  # 过孔层
        "3/0": 2   # 金属层 2
    }
    
    # 构建命令
    cmd = [
        sys.executable, script_path,
        "--gds-file", gds_file,
        "--output-dir", output_dir,
        "--patch-size", "5.0",
        "--patch-stride", "2.5"
    ]
    
    # 添加层映射参数
    for layer_str, idx in layer_mapping.items():
        cmd.extend(["--layer-mapping", f"{layer_str}:{idx}"])
    
    print(f"运行预处理命令: {' '.join(cmd)}")
    
    # 执行命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("预处理成功完成!")
        print("输出:")
        print(result.stdout)
    else:
        print("预处理失败!")
        print("错误:")
        print(result.stderr)

def main():
    """主函数"""
    # 定义路径
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    gds_file = os.path.join(examples_dir, "simple_layout.gds")
    output_dir = os.path.join(examples_dir, "processed_data")
    
    # 创建 GDS 文件
    create_simple_gds(gds_file)
    
    # 预处理数据
    preprocess_sample_data(gds_file, output_dir)
    
    print("\n示例数据生成完成!")
    print(f"GDS 文件: {gds_file}")
    print(f"处理后的数据: {output_dir}")

if __name__ == "__main__":
    main()
