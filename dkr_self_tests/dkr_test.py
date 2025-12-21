"""
单独分析 race 字段的详细情况
查看为什么结果中还有数字
"""
import pandas as pd
import numpy as np
from collections import Counter


def analyze_race_detailed(train_tsv_path: str = "datasets/news_personality/train.tsv"):
    """
    详细分析 race 字段
    
    Args:
        train_tsv_path: train.tsv文件路径
    """
    print("="*80)
    print("Race 字段详细分析")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv(train_tsv_path, sep='\t')
    print(f"\n总样本数: {len(df)}")
    
    # 获取 race 列
    race_data = df['race'].copy()
    
    # 1. 查看原始值
    print("\n" + "="*80)
    print("1. 原始值类型和示例")
    print("="*80)
    print(f"数据类型: {race_data.dtype}")
    print(f"\n前20个原始值:")
    print(race_data.head(20).tolist())
    
    # 2. 查看所有唯一值
    print("\n" + "="*80)
    print("2. 所有唯一值及其类型")
    print("="*80)
    unique_values = race_data.unique()
    print(f"唯一值数量: {len(unique_values)}")
    print(f"\n唯一值列表:")
    for i, val in enumerate(sorted(unique_values, key=lambda x: str(x))):
        val_type = type(val).__name__
        print(f"  [{i+1}] 值: {repr(val):<15s} 类型: {val_type}")
    
    # 3. 统计每个值的频数（原始）
    print("\n" + "="*80)
    print("3. 原始值频数统计")
    print("="*80)
    value_counts = race_data.value_counts().sort_index()
    print(f"{'原始值':<20s} {'类型':<15s} {'频数':<10s} {'百分比':<10s}")
    print("-"*60)
    total = len(race_data)
    for value, count in value_counts.items():
        val_type = type(value).__name__
        pct = count / total * 100
        print(f"{repr(value):<20s} {val_type:<15s} {count:>8d} {pct:>8.1f}%")
    
    # 4. 值类型转换分析
    print("\n" + "="*80)
    print("4. 值类型转换分析")
    print("="*80)
    
    print("\n尝试将每个值转换为数字（保留原始值）:")
    print(f"{'原始值':<20s} {'原始类型':<15s} {'转换后数字':<15s} {'转换状态':<15s}")
    print("-"*70)
    
    for value in sorted(unique_values, key=lambda x: str(x)):
        original_type = type(value).__name__
        converted_num = None
        status = ""
        
        try:
            # 如果是字符串 'unknown'
            if str(value).lower() == 'unknown':
                converted_num = 'unknown'
                status = "保留为unknown"
            # 如果是数字字符串或数字
            elif str(value).replace('.', '').isdigit():
                converted_num = int(float(value))
                status = "✓ 成功"
            # 如果是数字类型
            elif isinstance(value, (int, float)) and not pd.isna(value):
                converted_num = int(value)
                status = "✓ 成功"
            else:
                converted_num = value
                status = "保持原值"
        except Exception as e:
            converted_num = f"错误: {e}"
            status = "✗ 失败"
        
        print(f"{repr(value):<20s} {original_type:<15s} {str(converted_num):<15s} {status:<15s}")
    
    # 5. 处理后的分布统计（保留数字）
    print("\n" + "="*80)
    print("5. 分布统计（保留数字值）")
    print("="*80)
    
    # 处理数据
    field_data = race_data.copy()
    field_data = field_data.replace('unknown', 'unknown')
    
    value_counts_processed = field_data.value_counts().sort_index()
    total_count = len(field_data)
    
    print(f"\n{'值':<20s} {'频数':<10s} {'百分比':<10s} {'累计百分比':<10s}")
    print("-"*60)
    
    cumulative = 0
    for value, count in value_counts_processed.items():
        pct = count / total_count * 100
        cumulative += pct
        print(f"{repr(value):<20s} {count:>8d} {pct:>8.1f}% {cumulative:>8.1f}%")
    
    # 6. 数字值统计
    print("\n" + "="*80)
    print("6. 数字值统计")
    print("="*80)
    
    numeric_values = []
    for value in unique_values:
        try:
            if str(value).replace('.', '').isdigit():
                numeric_values.append(int(float(value)))
            elif isinstance(value, (int, float)) and not pd.isna(value):
                numeric_values.append(int(value))
        except:
            pass
    
    if numeric_values:
        numeric_counter = Counter(numeric_values)
        print(f"\n出现的数字值及其频数:")
        for num_val, count in sorted(numeric_counter.items()):
            pct = count / total * 100
            print(f"  {num_val}: {count} 次 ({pct:.1f}%)")
    else:
        print("\n未发现数字值")
    
    # 7. 总结
    print("\n" + "="*80)
    print("7. 总结")
    print("="*80)
    print(f"总样本数: {len(df)}")
    print(f"唯一值数量: {len(unique_values)}")
    numeric_count = len([v for v in unique_values if str(v).replace('.', '').isdigit() or (isinstance(v, (int, float)) and not pd.isna(v))])
    unknown_count = len([v for v in unique_values if str(v).lower() == 'unknown'])
    print(f"数字值数量: {numeric_count}")
    print(f"unknown值数量: {unknown_count}")
    print(f"\n所有值都保留原始格式，不做任何映射转换")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='详细分析 race 字段')
    parser.add_argument('--train_file', type=str, 
                       default='datasets/news_personality/train.tsv',
                       help='train.tsv文件路径')
    
    args = parser.parse_args()
    
    analyze_race_detailed(args.train_file)

