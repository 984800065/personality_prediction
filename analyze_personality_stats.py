"""
分析train.tsv中5种人格特征的统计信息
包括：取值范围（最小值、最大值）、均值、方差
以及人口统计学特征（gender, education, race, age, income）的完整分析
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional


def analyze_personality_stats(train_tsv_path: str = "datasets/news_personality/train.tsv"):
    """
    分析train.tsv中5种人格特征的统计信息
    
    Args:
        train_tsv_path: train.tsv文件路径
    """
    # 读取数据
    print(f"正在读取数据文件: {train_tsv_path}")
    df = pd.read_csv(train_tsv_path, sep='\t')
    print(f"总样本数: {len(df)}")
    
    # 5个人格维度列名
    personality_columns = [
        'personality_conscientiousness',  # 尽责性
        'personality_openess',            # 开放性
        'personality_extraversion',        # 外向性
        'personality_agreeableness',       # 宜人性
        'personality_stability'            # 稳定性（神经质的反向）
    ]
    
    # 中文名称映射
    chinese_names = {
        'personality_conscientiousness': '尽责性 (Conscientiousness)',
        'personality_openess': '开放性 (Openness)',
        'personality_extraversion': '外向性 (Extraversion)',
        'personality_agreeableness': '宜人性 (Agreeableness)',
        'personality_stability': '稳定性 (Stability)'
    }
    
    # 检查列是否存在
    missing_columns = [col for col in personality_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: 以下列不存在: {missing_columns}")
        return
    
    # 提取人格维度数据
    personality_df = df[personality_columns].copy()
    
    # 处理'unknown'值：替换为NaN
    personality_df = personality_df.replace('unknown', pd.NA)
    
    # 转换为数值类型
    for col in personality_columns:
        personality_df[col] = pd.to_numeric(personality_df[col], errors='coerce')
    
    # 统计每个维度的有效样本数
    print("\n" + "="*80)
    print("各维度有效样本数统计:")
    print("="*80)
    for col in personality_columns:
        valid_count = personality_df[col].notna().sum()
        total_count = len(personality_df)
        print(f"{chinese_names[col]:30s}: {valid_count:4d} / {total_count:4d} ({valid_count/total_count*100:.1f}%)")
    
    # 计算统计信息
    print("\n" + "="*80)
    print("各维度统计信息:")
    print("="*80)
    print(f"{'维度':<35s} {'最小值':<12s} {'最大值':<12s} {'均值':<12s} {'方差':<12s} {'标准差':<12s}")
    print("-"*80)
    
    stats_results = {}
    
    for col in personality_columns:
        # 获取有效数据（排除NaN）
        valid_data = personality_df[col].dropna()
        
        if len(valid_data) == 0:
            print(f"{chinese_names[col]:<35s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s}")
            continue
        
        # 计算统计量
        min_val = valid_data.min()
        max_val = valid_data.max()
        mean_val = valid_data.mean()
        var_val = valid_data.var()
        std_val = valid_data.std()
        
        # 保存结果
        stats_results[col] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'variance': var_val,
            'std': std_val,
            'valid_count': len(valid_data)
        }
        
        # 打印结果
        print(f"{chinese_names[col]:<35s} {min_val:>11.4f} {max_val:>11.4f} {mean_val:>11.4f} {var_val:>11.4f} {std_val:>11.4f}")
    
    # 额外统计：整体分布
    print("\n" + "="*80)
    print("整体统计摘要:")
    print("="*80)
    
    all_valid_data = []
    for col in personality_columns:
        valid_data = personality_df[col].dropna()
        all_valid_data.extend(valid_data.tolist())
    
    if all_valid_data:
        all_array = np.array(all_valid_data)
        print(f"所有维度合并后的统计:")
        print(f"  最小值: {all_array.min():.4f}")
        print(f"  最大值: {all_array.max():.4f}")
        print(f"  均值: {all_array.mean():.4f}")
        print(f"  方差: {all_array.var():.4f}")
        print(f"  标准差: {all_array.std():.4f}")
    
    # 保存结果到文件
    output_file = "personality_stats.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("人格特征统计信息\n")
        f.write("="*80 + "\n\n")
        f.write(f"数据文件: {train_tsv_path}\n")
        f.write(f"总样本数: {len(df)}\n\n")
        
        f.write("各维度统计信息:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'维度':<35s} {'最小值':<12s} {'最大值':<12s} {'均值':<12s} {'方差':<12s} {'标准差':<12s}\n")
        f.write("-"*80 + "\n")
        
        for col in personality_columns:
            if col in stats_results:
                stats = stats_results[col]
                f.write(f"{chinese_names[col]:<35s} {stats['min']:>11.4f} {stats['max']:>11.4f} "
                        f"{stats['mean']:>11.4f} {stats['variance']:>11.4f} {stats['std']:>11.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("详细统计:\n")
        f.write("="*80 + "\n")
        for col in personality_columns:
            if col in stats_results:
                stats = stats_results[col]
                f.write(f"\n{chinese_names[col]}:\n")
                f.write(f"  有效样本数: {stats['valid_count']}\n")
                f.write(f"  取值范围: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
                f.write(f"  均值: {stats['mean']:.4f}\n")
                f.write(f"  方差: {stats['variance']:.4f}\n")
                f.write(f"  标准差: {stats['std']:.4f}\n")
    
    print(f"\n统计结果已保存到: {output_file}")
    
    return stats_results


def analyze_demographic_features(
    train_tsv_path: str = "datasets/news_personality/train.tsv"
) -> Dict[str, Any]:
    """
    分析人口统计学特征（gender, education, race, age, income）
    
    Args:
        train_tsv_path: train.tsv文件路径
    
    Returns:
        包含所有分析结果的字典
    """
    # 读取数据
    print(f"\n{'='*80}")
    print("人口统计学特征分析")
    print(f"{'='*80}")
    print(f"正在读取数据文件: {train_tsv_path}")
    df = pd.read_csv(train_tsv_path, sep='\t')
    print(f"总样本数: {len(df)}")
    
    # 定义字段名称（用于输出）
    field_mapping = {
        'gender': {
            'name': '性别 (Gender)'
        },
        'education': {
            'name': '教育程度 (Education)'
        },
        'race': {
            'name': '种族 (Race)'
        },
        'age': {
            'name': '年龄 (Age)'
        },
        'income': {
            'name': '收入 (Income)'
        }
    }
    
    # 5个人格维度列名
    personality_columns = [
        'personality_conscientiousness',
        'personality_openess',
        'personality_extraversion',
        'personality_agreeableness',
        'personality_stability'
    ]
    
    results = {}
    
    # 分析每个字段
    for field in ['gender', 'education', 'race', 'age', 'income']:
        print(f"\n{'-'*80}")
        print(f"【{field_mapping[field]['name']}】")
        print(f"{'-'*80}")
        
        if field in ['age', 'income']:
            # 数值型字段分析
            field_data = df[field].copy()
            
            # 处理unknown值
            field_data = field_data.replace('unknown', pd.NA)
            field_data = pd.to_numeric(field_data, errors='coerce')
            
            # 基本统计
            valid_count = field_data.notna().sum()
            missing_count = field_data.isna().sum()
            valid_rate = valid_count / len(df) * 100
            
            print(f"\n📊 基本统计:")
            print(f"  有效样本数: {valid_count} / {len(df)} ({valid_rate:.1f}%)")
            print(f"  缺失样本数: {missing_count} ({100-valid_rate:.1f}%)")
            
            if valid_count > 0:
                print(f"\n📈 数值统计:")
                print(f"  最小值: {field_data.min():.2f}")
                print(f"  最大值: {field_data.max():.2f}")
                print(f"  均值: {field_data.mean():.2f}")
                print(f"  中位数: {field_data.median():.2f}")
                print(f"  标准差: {field_data.std():.2f}")
                print(f"  25%分位数: {field_data.quantile(0.25):.2f}")
                print(f"  75%分位数: {field_data.quantile(0.75):.2f}")
                
                # 分组统计（如果是age，按年龄段；如果是income，按收入区间）
                if field == 'age':
                    print(f"\n📋 年龄段分布:")
                    age_bins = [0, 25, 30, 35, 40, 45, 50, 100]
                    age_labels = ['<25', '25-29', '30-34', '35-39', '40-44', '45-49', '50+']
                    age_groups = pd.cut(field_data, bins=age_bins, labels=age_labels, include_lowest=True)
                    age_counts = age_groups.value_counts().sort_index()
                    for label, count in age_counts.items():
                        pct = count / valid_count * 100
                        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
                
                elif field == 'income':
                    print(f"\n📋 收入区间分布:")
                    income_bins = [0, 30000, 50000, 70000, 90000, 120000, float('inf')]
                    income_labels = ['<30K', '30K-50K', '50K-70K', '70K-90K', '90K-120K', '120K+']
                    income_groups = pd.cut(field_data, bins=income_bins, labels=income_labels, include_lowest=True)
                    income_counts = income_groups.value_counts().sort_index()
                    for label, count in income_counts.items():
                        pct = count / valid_count * 100
                        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%)")
            
            results[field] = {
                'type': 'numeric',
                'valid_count': valid_count,
                'missing_count': missing_count,
                'stats': {
                    'min': float(field_data.min()) if valid_count > 0 else None,
                    'max': float(field_data.max()) if valid_count > 0 else None,
                    'mean': float(field_data.mean()) if valid_count > 0 else None,
                    'median': float(field_data.median()) if valid_count > 0 else None,
                    'std': float(field_data.std()) if valid_count > 0 else None,
                } if valid_count > 0 else None
            }
            
        else:
            # 分类型字段分析
            field_data = df[field].copy()
            
            # 处理unknown值
            field_data = field_data.replace('unknown', 'unknown')
            
            # 统计频数
            value_counts = field_data.value_counts().sort_index()
            total_count = len(field_data)
            
            print(f"\n📊 分布统计:")
            print(f"{'值':<20s} {'频数':<10s} {'百分比':<10s} {'累计百分比':<10s}")
            print(f"{'-'*50}")
            
            cumulative = 0
            distribution = {}
            for value, count in value_counts.items():
                pct = count / total_count * 100
                cumulative += pct
                # 直接显示原始值，不做任何映射
                value_label = str(value)
                print(f"{value_label:<20s} {count:>8d} {pct:>8.1f}% {cumulative:>8.1f}%")
                distribution[str(value)] = {'count': int(count), 'percentage': float(pct)}
            
            # 缺失值统计
            unknown_count = (field_data == 'unknown').sum() if 'unknown' in value_counts else 0
            valid_count = total_count - unknown_count
            
            print(f"\n📈 汇总:")
            print(f"  有效样本数: {valid_count} / {total_count} ({valid_count/total_count*100:.1f}%)")
            print(f"  缺失/未知: {unknown_count} ({unknown_count/total_count*100:.1f}%)")
            
            results[field] = {
                'type': 'categorical',
                'valid_count': valid_count,
                'missing_count': unknown_count,
                'distribution': distribution
            }
    
    # 分析各字段与人格特征的关系
    print(f"\n{'='*80}")
    print("人口统计学特征与人格特征的关系分析")
    print(f"{'='*80}")
    
    # 准备人格特征数据
    personality_df = df[personality_columns].copy()
    personality_df = personality_df.replace('unknown', pd.NA)
    for col in personality_columns:
        personality_df[col] = pd.to_numeric(personality_df[col], errors='coerce')
    
    # 分析每个字段与人格特征的关系
    for field in ['gender', 'education', 'race', 'age', 'income']:
        print(f"\n{'-'*80}")
        print(f"【{field_mapping[field]['name']} 与人格特征的关系】")
        print(f"{'-'*80}")
        
        field_data = df[field].copy()
        field_data = field_data.replace('unknown', pd.NA)
        
        if field in ['age', 'income']:
            # 数值型：计算相关系数
            field_data = pd.to_numeric(field_data, errors='coerce')
            
            print(f"\n📊 相关系数 (Pearson):")
            print(f"{'人格维度':<35s} {'相关系数':<12s} {'解释':<20s}")
            print(f"{'-'*70}")
            
            for col in personality_columns:
                # 计算相关系数（只使用两个字段都有效的样本）
                valid_mask = field_data.notna() & personality_df[col].notna()
                if valid_mask.sum() > 10:  # 至少需要10个有效样本
                    corr = field_data[valid_mask].corr(personality_df[col][valid_mask])
                    # 解释相关性强度
                    abs_corr = abs(corr)
                    if abs_corr < 0.1:
                        strength = "极弱"
                    elif abs_corr < 0.3:
                        strength = "弱"
                    elif abs_corr < 0.5:
                        strength = "中等"
                    elif abs_corr < 0.7:
                        strength = "强"
                    else:
                        strength = "极强"
                    
                    print(f"{col:<35s} {corr:>11.4f} {strength:<20s}")
                else:
                    print(f"{col:<35s} {'N/A':<12s} {'样本不足':<20s}")
        
        else:
            # 分类型：按组统计均值
            field_data = field_data.astype(str)  # 转换为字符串以便分组
            
            print(f"\n📊 各分组的人格特征均值:")
            
            # 获取所有有效值（排除unknown和NaN）
            valid_values = field_data[field_data.notna() & (field_data != 'unknown') & (field_data != 'nan')].unique()
            # 排序：尝试转换为数字排序，否则按字符串排序
            try:
                valid_values = sorted([v for v in valid_values if v], key=lambda x: float(x) if str(x).replace('.', '').isdigit() else 0)
            except:
                valid_values = sorted([v for v in valid_values if v])
            
            if len(valid_values) > 0:
                # 表头
                header = f"{'分组':<20s}"
                for col in personality_columns:
                    header += f" {col.split('_')[-1][:8]:<12s}"
                print(header)
                print(f"{'-'*80}")
                
                # 每个分组
                for value in valid_values:
                    mask = (field_data == str(value))
                    if mask.sum() > 0:
                        # 直接显示原始值，不做任何映射
                        value_label = str(value)
                        row = f"{value_label:<20s}"
                        
                        for col in personality_columns:
                            group_data = personality_df[col][mask]
                            valid_group_data = group_data.dropna()
                            if len(valid_group_data) > 0:
                                mean_val = valid_group_data.mean()
                                row += f" {mean_val:>11.4f}"
                            else:
                                row += f" {'N/A':>11s}"
                        print(row)
    
    # 保存结果到文件
    output_file = "demographic_stats.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("人口统计学特征分析报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"数据文件: {train_tsv_path}\n")
        f.write(f"总样本数: {len(df)}\n\n")
        
        # 写入详细结果
        for field in ['gender', 'education', 'race', 'age', 'income']:
            f.write("\n" + "="*80 + "\n")
            f.write(f"{field_mapping[field]['name']}\n")
            f.write("="*80 + "\n\n")
            
            if field in ['age', 'income']:
                stats = results[field]['stats']
                if stats:
                    f.write(f"有效样本数: {results[field]['valid_count']}\n")
                    f.write(f"缺失样本数: {results[field]['missing_count']}\n\n")
                    f.write("数值统计:\n")
                    f.write(f"  最小值: {stats['min']:.2f}\n")
                    f.write(f"  最大值: {stats['max']:.2f}\n")
                    f.write(f"  均值: {stats['mean']:.2f}\n")
                    f.write(f"  中位数: {stats['median']:.2f}\n")
                    f.write(f"  标准差: {stats['std']:.2f}\n")
            else:
                f.write(f"有效样本数: {results[field]['valid_count']}\n")
                f.write(f"缺失样本数: {results[field]['missing_count']}\n\n")
                f.write("分布统计:\n")
                for value, info in results[field]['distribution'].items():
                    # 直接显示原始值，不做任何映射
                    value_label = str(value)
                    f.write(f"  {value_label}: {info['count']} ({info['percentage']:.1f}%)\n")
    
    print(f"\n{'='*80}")
    print(f"✅ 人口统计学特征分析完成！")
    print(f"📄 详细结果已保存到: {output_file}")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分析train.tsv中5种人格特征和人口统计学特征的统计信息')
    parser.add_argument('--train_file', type=str, 
                       default='datasets/news_personality/train.tsv',
                       help='train.tsv文件路径')
    parser.add_argument('--no-demographic', action='store_true',
                       help='跳过人口统计学特征分析')
    
    args = parser.parse_args()
    
    # 分析人格特征
    stats = analyze_personality_stats(args.train_file)
    
    # 分析人口统计学特征（默认启用）
    if not args.no_demographic:
        demographic_stats = analyze_demographic_features(args.train_file)

