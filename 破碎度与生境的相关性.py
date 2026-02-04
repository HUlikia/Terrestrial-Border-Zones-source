import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import os
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ===================== 路径与情景 =====================
lucc_dir = r"D:\SD_PLUS_Model\paper_result\landuse"
hq_dir = r"D:\SD_PLUS_Model\paper_result\quality_all"

scenarios = [
    "PES_2030", "ESP_2030", "SCS_2030",
    "PES_2040", "ESP_2040", "SCS_2040"
]

# ===================== 参数设置 =====================
grid_size = 1000  # 网格大小（米）
cell_size = 30  # 原始像元大小（米）
sample_rate = 0.02  # 采样率
grid_cells = int(grid_size / cell_size)  # 网格内的像元数


# ===================== 主要函数定义 =====================

def calculate_landscape_indices(lucc_array, forest_value=2):
    """
    计算网格内的景观指数
    返回: 破碎度指数(FI), 连通性指数(CI), 森林覆盖率(FCR)
    """
    # 创建森林二值掩膜
    forest_mask = (lucc_array == forest_value).astype(np.uint8)

    # 计算森林面积和覆盖率
    forest_area = np.sum(forest_mask)
    total_pixels = lucc_array.size
    forest_cover_ratio = forest_area / total_pixels if total_pixels > 0 else 0.0

    # 如果森林面积为0，返回默认值
    if forest_area == 0:
        return 0.0, 0.0, forest_cover_ratio

    # ========== 计算破碎度指数(FI) ==========
    edge_pixels = 0
    rows, cols = forest_mask.shape

    # 使用卷积核检测边缘
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if forest_mask[i, j] == 1:
                # 检查8邻域
                neighbors = forest_mask[i - 1:i + 2, j - 1:j + 2]
                if np.sum(neighbors) < 9:  # 如果不是所有邻域都是森林
                    edge_pixels += 1

    # 避免除以零
    if forest_area > 0:
        fragmentation_index = edge_pixels / forest_area
    else:
        fragmentation_index = 0.0

    # ========== 计算连通性指数(CI) ==========
    if forest_area > 0:
        max_perimeter = 4 * np.sqrt(forest_area)
        actual_perimeter = edge_pixels * 4
        if max_perimeter > 0:
            connectivity_index = 1 - (actual_perimeter / max_perimeter)
            connectivity_index = max(0.0, min(1.0, connectivity_index))
        else:
            connectivity_index = 0.0
    else:
        connectivity_index = 0.0

    return fragmentation_index, connectivity_index, forest_cover_ratio


def calculate_habitat_quality_in_grid(hq_array):
    """
    计算网格内平均生境质量
    """
    # 过滤掉无效值
    mask = (hq_array >= 0) & (hq_array <= 1)
    valid_hq = hq_array[mask]

    if len(valid_hq) > 0:
        return np.mean(valid_hq)
    return np.nan


def sample_grid_data(scenario):
    """
    在网格尺度采样数据
    """
    lucc_file = os.path.join(lucc_dir, f"{scenario}.tif")
    hq_file = os.path.join(hq_dir, f"{scenario}.tif")

    if not os.path.exists(lucc_file) or not os.path.exists(hq_file):
        return []

    results = []

    try:
        with rasterio.open(lucc_file) as lucc_src:
            lucc_width = lucc_src.width
            lucc_height = lucc_src.height

            total_grids_x = lucc_width // grid_cells
            total_grids_y = lucc_height // grid_cells

            if total_grids_x < 2 or total_grids_y < 2:
                return []

            total_possible_grids = total_grids_x * total_grids_y
            sample_count = max(50, int(total_possible_grids * sample_rate))

            attempts = 0
            max_attempts = sample_count * 3

            while len(results) < sample_count and attempts < max_attempts:
                attempts += 1

                start_x = np.random.randint(0, max(1, total_grids_x - 1)) * grid_cells
                start_y = np.random.randint(0, max(1, total_grids_y - 1)) * grid_cells

                try:
                    lucc_window = lucc_src.read(1,
                                                window=Window(start_x, start_y, grid_cells, grid_cells))

                    fi, ci, fcr = calculate_landscape_indices(lucc_window)

                    if (np.isfinite(fi) and np.isfinite(ci) and np.isfinite(fcr) and
                            fi >= 0 and ci >= 0 and fcr >= 0):

                        with rasterio.open(hq_file) as hq_src:
                            hq_window = hq_src.read(1,
                                                    window=Window(start_x, start_y, grid_cells, grid_cells))
                            avg_hq = calculate_habitat_quality_in_grid(hq_window)

                            if np.isfinite(avg_hq) and avg_hq >= 0:
                                results.append({
                                    'scenario': scenario,
                                    'fragmentation_index': fi,
                                    'connectivity_index': ci,
                                    'forest_cover_ratio': fcr,
                                    'habitat_quality': avg_hq,
                                    'grid_x': start_x,
                                    'grid_y': start_y
                                })
                except:
                    continue

    except Exception as e:
        print(f"处理{scenario}时出错: {str(e)}")

    return results


def clean_correlation_data(x, y):
    """
    清洗相关性分析数据
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) > 0:
        valid_mask = (x_clean >= 0) & (x_clean <= 2) & (y_clean >= 0) & (y_clean <= 1)
        x_clean = x_clean[valid_mask]
        y_clean = y_clean[valid_mask]

    return x_clean, y_clean


def calculate_correlations(df):
    """
    计算相关性统计
    """
    correlation_results = []

    # 整体相关性
    fi_all = df['fragmentation_index'].values
    ci_all = df['connectivity_index'].values
    hq_all = df['habitat_quality'].values

    fi_clean, hq_fi_clean = clean_correlation_data(fi_all, hq_all)
    ci_clean, hq_ci_clean = clean_correlation_data(ci_all, hq_all)

    if len(fi_clean) > 10:
        fi_hq_corr, fi_hq_p = stats.pearsonr(fi_clean, hq_fi_clean)
    else:
        fi_hq_corr, fi_hq_p = np.nan, np.nan

    if len(ci_clean) > 10:
        ci_hq_corr, ci_hq_p = stats.pearsonr(ci_clean, hq_ci_clean)
    else:
        ci_hq_corr, ci_hq_p = np.nan, np.nan

    correlation_results.append({
        'group': 'ALL',
        'samples': len(df),
        'valid_fi_samples': len(fi_clean),
        'valid_ci_samples': len(ci_clean),
        'mean_fi': np.mean(fi_clean) if len(fi_clean) > 0 else np.nan,
        'mean_ci': np.mean(ci_clean) if len(ci_clean) > 0 else np.nan,
        'mean_hq': np.mean(hq_all[np.isfinite(hq_all)]),
        'fi_hq_corr': fi_hq_corr,
        'fi_hq_p_value': fi_hq_p,
        'ci_hq_corr': ci_hq_corr,
        'ci_hq_p_value': ci_hq_p,
        'fi_hq_significant': fi_hq_p < 0.05 if not np.isnan(fi_hq_p) else False,
        'ci_hq_significant': ci_hq_p < 0.05 if not np.isnan(ci_hq_p) else False
    })

    # 分情景相关性
    for scenario in scenarios:
        subset = df[df['scenario'] == scenario]

        if len(subset) > 10:
            fi_scene = subset['fragmentation_index'].values
            ci_scene = subset['connectivity_index'].values
            hq_scene = subset['habitat_quality'].values

            fi_clean_scene, hq_fi_clean_scene = clean_correlation_data(fi_scene, hq_scene)
            ci_clean_scene, hq_ci_clean_scene = clean_correlation_data(ci_scene, hq_scene)

            if len(fi_clean_scene) > 10:
                fi_hq_corr_scene, fi_hq_p_scene = stats.pearsonr(fi_clean_scene, hq_fi_clean_scene)
            else:
                fi_hq_corr_scene, fi_hq_p_scene = np.nan, np.nan

            if len(ci_clean_scene) > 10:
                ci_hq_corr_scene, ci_hq_p_scene = stats.pearsonr(ci_clean_scene, hq_ci_clean_scene)
            else:
                ci_hq_corr_scene, ci_hq_p_scene = np.nan, np.nan

            # 解释强度
            if not np.isnan(fi_hq_corr_scene):
                if fi_hq_corr_scene < -0.6:
                    fi_hq_strength = "强负相关"
                elif fi_hq_corr_scene < -0.3:
                    fi_hq_strength = "中度负相关"
                elif fi_hq_corr_scene < 0:
                    fi_hq_strength = "弱负相关"
                else:
                    fi_hq_strength = "正相关"
            else:
                fi_hq_strength = "无效"

            if not np.isnan(ci_hq_corr_scene):
                if ci_hq_corr_scene > 0.6:
                    ci_hq_strength = "强正相关"
                elif ci_hq_corr_scene > 0.3:
                    ci_hq_strength = "中度正相关"
                elif ci_hq_corr_scene > 0:
                    ci_hq_strength = "弱正相关"
                else:
                    ci_hq_strength = "负相关"
            else:
                ci_hq_strength = "无效"

            correlation_results.append({
                'group': scenario,
                'samples': len(subset),
                'valid_fi_samples': len(fi_clean_scene),
                'valid_ci_samples': len(ci_clean_scene),
                'mean_fi': np.mean(fi_clean_scene) if len(fi_clean_scene) > 0 else np.nan,
                'mean_ci': np.mean(ci_clean_scene) if len(ci_clean_scene) > 0 else np.nan,
                'mean_hq': np.mean(hq_scene[np.isfinite(hq_scene)]),
                'fi_hq_corr': fi_hq_corr_scene,
                'fi_hq_p_value': fi_hq_p_scene,
                'ci_hq_corr': ci_hq_corr_scene,
                'ci_hq_p_value': ci_hq_p_scene,
                'fi_hq_strength': fi_hq_strength,
                'ci_hq_strength': ci_hq_strength,
                'fi_hq_significant': fi_hq_p_scene < 0.05 if not np.isnan(fi_hq_p_scene) else False,
                'ci_hq_significant': ci_hq_p_scene < 0.05 if not np.isnan(ci_hq_p_scene) else False
            })

    return pd.DataFrame(correlation_results)


# ===================== 主程序 =====================
def main():
    """主函数：仅生成表格数据，不绘图"""

    print("开始数据采样与景观指数计算...")
    print("=" * 60)

    # 存储所有数据
    all_data = []

    # 为每个情景采样数据
    for scenario in scenarios:
        print(f"正在处理: {scenario}")
        data = sample_grid_data(scenario)
        if data:
            all_data.extend(data)
            print(f"  - 采集到{len(data)}个样本")
        else:
            print(f"  - 未采集到样本")

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("没有采集到有效数据！")
        return

    print(f"\n总共采集到{len(df)}个有效样本")

    # 创建输出目录
    output_dir = r"D:\SD_PLUS_Model\paper_result"
    os.makedirs(output_dir, exist_ok=True)

    # 保存原始数据
    raw_data_path = os.path.join(output_dir, "landscape_hq_raw_data.csv")
    df.to_csv(raw_data_path, index=False, encoding='utf-8-sig')
    print(f"\n原始数据已保存至: {raw_data_path}")

    # 计算相关性统计
    print("\n计算相关性统计...")
    correlations_df = calculate_correlations(df)

    # 保存相关性统计
    corr_stats_path = os.path.join(output_dir, "correlation_statistics.csv")
    correlations_df.to_csv(corr_stats_path, index=False, encoding='utf-8-sig')
    print(f"相关性统计已保存至: {corr_stats_path}")

    # 按情景汇总统计
    print(f"\n" + "=" * 60)
    print("各情景统计摘要")
    print("=" * 60)

    summary_stats = []
    for scenario in scenarios:
        subset = df[df['scenario'] == scenario]
        if len(subset) > 0:
            summary_stats.append({
                'Scenario': scenario,
                'Samples': len(subset),
                'Mean_FI': subset['fragmentation_index'].mean(),
                'Std_FI': subset['fragmentation_index'].std(),
                'Mean_CI': subset['connectivity_index'].mean(),
                'Std_CI': subset['connectivity_index'].std(),
                'Mean_HQ': subset['habitat_quality'].mean(),
                'Std_HQ': subset['habitat_quality'].std(),
                'Mean_FCR': subset['forest_cover_ratio'].mean()
            })

    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False))

    # 保存汇总统计
    summary_path = os.path.join(output_dir, "scenario_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n情景汇总统计已保存至: {summary_path}")

    # 生成论文表格格式
    print(f"\n" + "=" * 60)
    print("相关性分析结果（论文格式）")
    print("=" * 60)

    paper_table = []
    for scenario in scenarios:
        subset = correlations_df[correlations_df['group'] == scenario]
        if not subset.empty:
            row = subset.iloc[0]
            paper_table.append({
                '情景': scenario,
                '样本数': int(row['valid_fi_samples']),
                '平均破碎度(FI)': f"{row['mean_fi']:.3f}",
                '平均连通性(CI)': f"{row['mean_ci']:.3f}",
                '平均生境质量(HQ)': f"{row['mean_hq']:.3f}",
                'FI-HQ相关系数': f"{row['fi_hq_corr']:.3f}{'*' if row['fi_hq_significant'] else ''}",
                'FI-HQ相关强度': row['fi_hq_strength'],
                'CI-HQ相关系数': f"{row['ci_hq_corr']:.3f}{'*' if row['ci_hq_significant'] else ''}",
                'CI-HQ相关强度': row['ci_hq_strength']
            })

    paper_df = pd.DataFrame(paper_table)
    print(paper_df.to_string(index=False))

    # 保存论文格式表格
    paper_table_path = os.path.join(output_dir, "paper_correlation_table.csv")
    paper_df.to_csv(paper_table_path, index=False, encoding='utf-8-sig')
    print(f"\n论文格式表格已保存至: {paper_table_path}")

    # 输出关键结论
    print(f"\n" + "=" * 60)
    print("分析完成！关键文件路径：")
    print("=" * 60)
    print(f"1. 原始采样数据: {raw_data_path}")
    print(f"2. 相关性统计: {corr_stats_path}")
    print(f"3. 情景汇总统计: {summary_path}")
    print(f"4. 论文格式表格: {paper_table_path}")

    print(f"\n您可以使用这些CSV文件在Excel或其他软件中绘制散点图。")
    print(f"推荐绘图方式：")
    print(f"- 使用Excel的散点图功能")
    print(f"- 使用Python的matplotlib（但不在本代码中运行）")
    print(f"- 使用Origin、GraphPad Prism等专业绘图软件")


# ===================== 执行程序 =====================
if __name__ == "__main__":
    main()