"""
从20×20到50×50（步长5），ρ从0.10到0.35（步长0.01），每个配置200局
支持断点重续，每个棋盘画一张ρ-胜率关系图
优化后算法（跳过大棋盘子集枚举 + frontier-only MC）
"""
import sys, time, csv, os
from pathlib import Path
sys.path.insert(0, '.')
from run_cuda_experiments import GPUMemoryPool, run_super_merged_all

# === 参数 ===
BOARD_SIZES = list(range(20, 51, 5))  # 20, 25, 30, 35, 40, 45, 50
DENSITY_START = 0.10
DENSITY_END = 0.35
DENSITY_STEP = 0.01
TRIALS = 200
BASE_SEED = 2026
OUTPUT_CSV = "sweep_20x50_results.csv"
PLOT_DIR = "sweep_20x50_plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# === 断点重续：读取已有结果 ===
completed = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['rows']), float(row['density']))
            completed.add(key)
    print(f"断点重续：已有 {len(completed)} 个配置完成，跳过")
else:
    # 写表头
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "rows", "cols", "total_cells", "mines", "density",
            "trials", "wins", "win_rate", "ci95",
            "avg_steps", "std_steps"
        ])

# === 逐棋盘大小运行 ===
overall_start = time.perf_counter()
total_processed = 0

for size in BOARD_SIZES:
    rows, cols = size, size
    total = rows * cols

    # 构建该棋盘大小的所有密度配置（跳过已完成的）
    configs = []
    d = DENSITY_START
    while d <= DENSITY_END + 1e-9:
        if (rows, round(d, 4)) not in completed:
            mines = max(1, min(total - 1, int(round(total * d))))
            actual_density = mines / total
            configs.append({
                "rows": rows, "cols": cols, "total_cells": total,
                "mines": mines, "density": actual_density, "ar": 1.0,
                "trials": TRIALS,
            })
        d += DENSITY_STEP

    if not configs:
        print(f"{size}×{size}: 全部完成，跳过")
        continue

    n_games = sum(c["trials"] for c in configs)
    total_processed += n_games

    print(f"\n{'='*50}")
    print(f"{size}×{size}: {len(configs)} 个密度 × {TRIALS} 局 = {n_games} 局")
    print(f"{'='*50}")

    # 分配显存池（按该棋盘大小的最大棋盘单元格数）
    gpu_pool = GPUMemoryPool(n_games * 2, total)

    t0 = time.perf_counter()
    results = run_super_merged_all(configs, BASE_SEED, gpu_pool)
    elapsed = time.perf_counter() - t0

    gpu_pool.free()

    print(f"  耗时: {elapsed:.2f}s ({n_games/elapsed:.0f} 局/秒)")

    # 追加写入 CSV
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for cfg, r in zip(configs, results):
            writer.writerow([
                cfg["rows"], cfg["cols"], cfg["total_cells"],
                cfg["mines"], f"{cfg['density']:.4f}",
                TRIALS, r["won"],
                f"{r['success_rate']:.6f}", f"{r['success_rate_ci95']:.6f}",
                f"{r['avg_steps']:.2f}", f"{r['std_steps']:.2f}",
            ])
    print(f"  结果已保存到 {OUTPUT_CSV}")

overall_elapsed = time.perf_counter() - overall_start
print(f"\n{'='*50}")
print(f"全部完成！共处理 {total_processed} 局")
print(f"总耗时: {overall_elapsed:.1f}s")
print(f"平均速度: {total_processed/overall_elapsed:.0f} 局/秒")
print(f"{'='*50}")

# === 绘图 ===
print(f"\n正在绘图...")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取所有结果
data = {}
with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = int(row['rows'])
        if key not in data:
            data[key] = {"density": [], "win_rate": [], "ci95": []}
        data[key]["density"].append(float(row['density']))
        data[key]["win_rate"].append(float(row['win_rate']))
        data[key]["ci95"].append(float(row['ci95']))

# 高斯CDF拟合函数
def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

from math import erf

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, size in enumerate(sorted(data.keys())):
    ax = axes[idx]
    d = data[size]

    # 按密度排序
    order = np.argsort(d["density"])
    density_arr = np.array(d["density"])[order]
    wr_arr = np.array(d["win_rate"])[order]
    ci_arr = np.array(d["ci95"])[order]

    # 绘图：数据点 + 误差条
    ax.errorbar(density_arr, wr_arr * 100, yerr=ci_arr * 100,
                fmt='o-', capsize=3, markersize=4, linewidth=1.5,
                color='#2196F3', ecolor='#BBDEFB', label='实测')

    # 拟合高斯CDF（只对非0%非100%的点拟合）
    valid = (wr_arr > 0.01) & (wr_arr < 0.99)
    if np.sum(valid) >= 3:
        try:
            p0 = [density_arr[valid][np.argmin(np.abs(wr_arr[valid] - 0.5))], 0.03]
            popt, _ = curve_fit(gaussian_cdf, density_arr[valid], wr_arr[valid],
                               p0=p0, maxfev=5000)
            smooth_x = np.linspace(density_arr[0], density_arr[-1], 200)
            smooth_y = gaussian_cdf(smooth_x, *popt)
            ax.plot(smooth_x, smooth_y * 100, '--', color='#FF5722',
                   linewidth=2, alpha=0.7,
                   label=f'高斯拟合 μ={popt[0]:.3f} σ={popt[1]:.3f}')
        except Exception as e:
            pass

    ax.set_xlabel('地雷密度 ρ', fontsize=11)
    ax.set_ylabel('胜率 (%)', fontsize=11)
    ax.set_title(f'{size}×{size} ({size*size} 格)', fontsize=13, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

# 隐藏多余的子图
for idx in range(len(data), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('不同棋盘大小的地雷密度-胜率关系 (MC算法, 200局/配置)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

plot_path = os.path.join(PLOT_DIR, "density_vs_winrate_all.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"组合图已保存: {plot_path}")

# 每个棋盘单独一张图
for size in sorted(data.keys()):
    fig, ax = plt.subplots(figsize=(8, 5))
    d = data[size]
    order = np.argsort(d["density"])
    density_arr = np.array(d["density"])[order]
    wr_arr = np.array(d["win_rate"])[order]
    ci_arr = np.array(d["ci95"])[order]

    ax.errorbar(density_arr, wr_arr * 100, yerr=ci_arr * 100,
                fmt='o-', capsize=4, markersize=5, linewidth=2,
                color='#2196F3', ecolor='#BBDEFB', label='实测 (200局)')

    valid = (wr_arr > 0.01) & (wr_arr < 0.99)
    if np.sum(valid) >= 3:
        try:
            p0 = [density_arr[valid][np.argmin(np.abs(wr_arr[valid] - 0.5))], 0.03]
            popt, _ = curve_fit(gaussian_cdf, density_arr[valid], wr_arr[valid],
                               p0=p0, maxfev=5000)
            smooth_x = np.linspace(density_arr[0], density_arr[-1], 200)
            smooth_y = gaussian_cdf(smooth_x, *popt)
            ax.plot(smooth_x, smooth_y * 100, '--', color='#FF5722',
                   linewidth=2.5, alpha=0.7,
                   label=f'高斯CDF拟合: μ={popt[0]:.3f}, σ={popt[1]:.3f}')
            # 标注关键点
            ax.axvline(x=popt[0], color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'μ={popt[0]:.3f}',
                       xy=(popt[0], 50), xytext=(popt[0]+0.01, 55),
                       fontsize=10, color='#FF5722', fontweight='bold')
        except Exception as e:
            pass

    ax.set_xlabel('地雷密度 ρ', fontsize=12)
    ax.set_ylabel('胜率 (%)', fontsize=12)
    ax.set_title(f'{size}×{size} 棋盘 (共{size*size}格)', fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    single_path = os.path.join(PLOT_DIR, f"density_vs_winrate_{size}x{size}.png")
    plt.savefig(single_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图已保存: {single_path}")

print(f"\n所有图已保存到 {PLOT_DIR}/ 目录")