import os
import sys
import time
import csv
import ctypes
import ctypes.util
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent
LIB_PATH = PROJECT_DIR / "lib"
DLL_FILE = LIB_PATH / "minesweeper.dll"

_cudart = None

def get_cudart():
    global _cudart
    if _cudart is not None:
        return _cudart
    cuda_paths = []
    cuda_home = os.environ.get("CUDA_PATH")
    if cuda_home:
        cuda_paths.append(cuda_home)
    cuda_paths.extend([
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    ])
    for cuda_path in cuda_paths:
        for subdir in ["bin", r"bin\x64"]:
            full_path = os.path.join(cuda_path, subdir)
            for dll_name in ["cudart64_13.dll", "cudart64_12.dll", "cudart64_130.dll", "cudart64_120.dll", "cudart64_11.dll"]:
                candidate = os.path.join(full_path, dll_name)
                if os.path.exists(candidate):
                    _cudart = ctypes.CDLL(candidate)
                    return _cudart
    lib_names = ["cudart64_13", "cudart64_12", "cudart64_130", "cudart64_120", "cudart", "libcudart.so"]
    for name in lib_names:
        try:
            _cudart = ctypes.CDLL(name)
            return _cudart
        except Exception:
            pass
    path = ctypes.util.find_library("cudart")
    if path:
        _cudart = ctypes.CDLL(path)
        return _cudart
    raise RuntimeError("CUDA runtime library not found.")

def cuda_malloc(size):
    cudart = get_cudart()
    dev_ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(dev_ptr), ctypes.c_size_t(size))
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed with error {err}")
    return dev_ptr.value

def cuda_free(dev_ptr):
    cudart = get_cudart()
    err = cudart.cudaFree(ctypes.c_void_p(dev_ptr))
    if err != 0:
        raise RuntimeError(f"cudaFree failed with error {err}")

def cuda_device_synchronize():
    cudart = get_cudart()
    err = cudart.cudaDeviceSynchronize()
    if err != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed with error {err}")

def get_cuda_lib():
    if not DLL_FILE.exists():
        raise RuntimeError(f"CUDA library not found at {DLL_FILE}. Run build.bat or build.ps1 first.")
    return ctypes.WinDLL(str(DLL_FILE))

class GPUMemoryPool:
    def __init__(self, total_threads, max_cells):
        self.total_threads = total_threads
        self.max_cells = max_cells
        self.int_work_size = total_threads * max_cells * 4
        self.inf_work_size = total_threads * max_cells * 4
        self.seen_work_size = total_threads * ((max_cells + 31) // 32) * 4
        self.results_size = total_threads * 4
        self._allocated = []

        self.d_results_win = cuda_malloc(self.results_size)
        self.d_results_steps = cuda_malloc(self.results_size)
        self.d_results_flags = cuda_malloc(self.results_size)
        self._allocated.extend([self.d_results_win, self.d_results_steps, self.d_results_flags])

        self.d_work_revealed = cuda_malloc(self.int_work_size)
        self.d_work_flagged = cuda_malloc(self.int_work_size)
        self.d_work_mine_cells = cuda_malloc(self.int_work_size)
        self.d_work_qr = cuda_malloc(self.int_work_size)
        self.d_work_qc = cuda_malloc(self.int_work_size)
        self.d_work_all = cuda_malloc(self.int_work_size)
        self.d_work_inf = cuda_malloc(self.inf_work_size)
        self.d_work_seen = cuda_malloc(self.seen_work_size)
        self._allocated.extend([
            self.d_work_revealed, self.d_work_flagged, self.d_work_mine_cells,
            self.d_work_qr, self.d_work_qc, self.d_work_all, self.d_work_inf,
            self.d_work_seen,
        ])

    def free(self):
        for ptr in self._allocated:
            try:
                cuda_free(ptr)
            except Exception:
                pass
        self._allocated.clear()

    def __del__(self):
        self.free()

def calc_gpu_threads(total_games, min_threads=64, max_threads_per_block=256, max_blocks=8192):
    threads_per_block = max_threads_per_block
    # 向下取整到 threads_per_block 的倍数，确保不超过 total_games
    num_blocks = max(1, total_games // threads_per_block)
    # 如果有余数，增加一个 block
    if total_games % threads_per_block != 0:
        num_blocks += 1
    # 但确保不超过 max_blocks
    num_blocks = min(num_blocks, max_blocks)
    # 最终 threads 不应该小于 min_threads
    total_threads = num_blocks * threads_per_block
    if total_threads < min_threads:
        num_blocks = (min_threads + threads_per_block - 1) // threads_per_block
        total_threads = num_blocks * threads_per_block
    return threads_per_block, num_blocks

def run_super_merged_all(
    all_configs, base_seed, gpu_pool
):
    """
    超级合并：所有配置（所有棋盘大小+所有密度）一次kernel launch跑完。
    all_configs: list of dicts with keys: rows, cols, mines, trials, density, ar, total_cells
    """
    num_configs = len(all_configs)
    total_games = sum(cfg["trials"] for cfg in all_configs)

    lib = get_cuda_lib()
    lib.launch_super_merged_minesweeper_kernel.argtypes = [
        ctypes.c_int, ctypes.c_int,  # num_blocks, threads_per_block
        ctypes.c_int,  # num_configs
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # rows, cols, mines
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # seeds, prefix_games, trials, offsets
        ctypes.c_int,  # total_games
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # revealed, flagged, mine_cells
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # qr, qc, all, inf
        ctypes.c_void_p,  # seen
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # out_win, out_steps, out_flags
    ]
    lib.launch_super_merged_minesweeper_kernel.restype = None

    h_rows = np.array([cfg["rows"] for cfg in all_configs], dtype=np.int32)
    h_cols = np.array([cfg["cols"] for cfg in all_configs], dtype=np.int32)
    h_mines = np.array([cfg["mines"] for cfg in all_configs], dtype=np.int32)
    h_seeds = np.array([base_seed] * num_configs, dtype=np.int32)
    h_gpc = np.array([cfg["trials"] for cfg in all_configs], dtype=np.int32)
    h_trials = h_gpc  # 每个配置的 trials

    # h_offsets: 每个配置在 pool 中的起始偏移量（以 int32 为单位）
    h_offsets = np.zeros(num_configs, dtype=np.int32)
    h_offsets[0] = 0
    for i in range(1, num_configs):
        h_offsets[i] = h_offsets[i-1] + all_configs[i-1]["rows"] * all_configs[i-1]["cols"] * all_configs[i-1]["trials"]

    h_prefix_games = np.cumsum(h_gpc).astype(np.int32)

    tpb, nb = calc_gpu_threads(total_games)

    d_rows = cuda_malloc(h_rows.nbytes)
    d_cols = cuda_malloc(h_cols.nbytes)
    d_mines = cuda_malloc(h_mines.nbytes)
    d_seeds = cuda_malloc(h_seeds.nbytes)
    d_prefix_games = cuda_malloc(h_prefix_games.nbytes)
    d_trials = cuda_malloc(h_trials.nbytes)
    d_offsets = cuda_malloc(h_offsets.nbytes)

    cudart = get_cudart()
    cudart.cudaMemcpy(ctypes.c_void_p(d_rows), h_rows.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_rows.nbytes), ctypes.c_int(1))
    cudart.cudaMemcpy(ctypes.c_void_p(d_cols), h_cols.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_cols.nbytes), ctypes.c_int(1))
    cudart.cudaMemcpy(ctypes.c_void_p(d_mines), h_mines.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_mines.nbytes), ctypes.c_int(1))
    cudart.cudaMemcpy(ctypes.c_void_p(d_seeds), h_seeds.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_seeds.nbytes), ctypes.c_int(1))
    cudart.cudaMemcpy(ctypes.c_void_p(d_prefix_games), h_prefix_games.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_prefix_games.nbytes), ctypes.c_int(1))
    cudart.cudaMemcpy(ctypes.c_void_p(d_trials), h_trials.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_trials.nbytes), ctypes.c_int(1))
    cudart.cudaMemcpy(ctypes.c_void_p(d_offsets), h_offsets.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(h_offsets.nbytes), ctypes.c_int(1))

    lib.launch_super_merged_minesweeper_kernel(
        nb, tpb,  # 注意：calc_gpu_threads 返回 (threads_per_block, num_blocks)，CUDA 是 <<<grid_dim, block_dim>>>
        num_configs,
        ctypes.c_void_p(d_rows), ctypes.c_void_p(d_cols), ctypes.c_void_p(d_mines),
        ctypes.c_void_p(d_seeds), ctypes.c_void_p(d_prefix_games), ctypes.c_void_p(d_trials), ctypes.c_void_p(d_offsets),
        total_games,
        ctypes.c_void_p(gpu_pool.d_work_revealed),
        ctypes.c_void_p(gpu_pool.d_work_flagged),
        ctypes.c_void_p(gpu_pool.d_work_mine_cells),
        ctypes.c_void_p(gpu_pool.d_work_qr),
        ctypes.c_void_p(gpu_pool.d_work_qc),
        ctypes.c_void_p(gpu_pool.d_work_all),
        ctypes.c_void_p(gpu_pool.d_work_inf),
        ctypes.c_void_p(gpu_pool.d_work_seen),
        ctypes.c_void_p(gpu_pool.d_results_win),
        ctypes.c_void_p(gpu_pool.d_results_steps),
        ctypes.c_void_p(gpu_pool.d_results_flags),
    )

    cuda_device_synchronize()

    h_win = np.zeros(total_games, dtype=np.int32)
    h_steps = np.zeros(total_games, dtype=np.int32)
    h_flags = np.zeros(total_games, dtype=np.int32)

    cudart.cudaMemcpy(
        h_win.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(gpu_pool.d_results_win),
        ctypes.c_size_t(total_games * 4),
        ctypes.c_int(2),
    )
    cudart.cudaMemcpy(
        h_steps.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(gpu_pool.d_results_steps),
        ctypes.c_size_t(total_games * 4),
        ctypes.c_int(2),
    )
    cudart.cudaMemcpy(
        h_flags.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(gpu_pool.d_results_flags),
        ctypes.c_size_t(total_games * 4),
        ctypes.c_int(2),
    )

    cuda_free(d_rows)
    cuda_free(d_cols)
    cuda_free(d_mines)
    cuda_free(d_seeds)
    cuda_free(d_prefix_games)
    cuda_free(d_offsets)

    offsets = np.concatenate([[0], np.cumsum(h_gpc[:-1])])
    results = []
    for i, cfg in enumerate(all_configs):
        start = offsets[i]
        end = start + cfg["trials"]
        wins_arr = h_win[start:end].astype(np.float64)
        steps_arr = h_steps[start:end].astype(np.float64)
        flags_arr = h_flags[start:end].astype(np.float64)
        trials = cfg["trials"]

        wins = int(wins_arr.sum())
        p = wins / trials if trials > 0 else 0.0

        # 胜率的高斯近似（二项分布 CLT）
        se_p = np.sqrt(p * (1 - p) / trials) if trials > 1 else 0.0
        ci95_p = 1.96 * se_p

        # 步数分布
        mean_steps = float(steps_arr.mean())
        std_steps = float(steps_arr.std(ddof=1)) if trials > 1 else 0.0
        se_steps = std_steps / np.sqrt(trials) if trials > 1 else 0.0
        ci95_steps = 1.96 * se_steps

        # 标旗数分布
        mean_flags = float(flags_arr.mean())
        std_flags = float(flags_arr.std(ddof=1)) if trials > 1 else 0.0
        se_flags = std_flags / np.sqrt(trials) if trials > 1 else 0.0
        ci95_flags = 1.96 * se_flags

        results.append({
            "won": wins,
            "lost": trials - wins,
            "total_games": trials,
            "success_rate": p,
            "success_rate_se": se_p,
            "success_rate_ci95": ci95_p,
            "avg_steps": mean_steps,
            "std_steps": std_steps,
            "steps_se": se_steps,
            "steps_ci95": ci95_steps,
            "avg_flags": mean_flags,
            "std_flags": std_flags,
            "flags_se": se_flags,
            "flags_ci95": ci95_flags,
            "steps_arr": steps_arr,
            "flags_arr": flags_arr,
        })

    return results

def make_board_shape(total_cells, aspect_ratio):
    best_rows, best_cols = 2, max(2, total_cells // 2)
    best_diff = float("inf")
    for rows in range(2, total_cells + 1):
        cols = int(round(total_cells / rows))
        if cols < 2:
            break
        actual = rows * cols
        actual_ratio = cols / rows
        diff = abs(actual - total_cells) + abs(actual_ratio - aspect_ratio) * total_cells * 0.1
        if diff < best_diff:
            best_diff = diff
            best_rows, best_cols = rows, cols
    return best_rows, best_cols

def run_full_experiments(
    aspect_ratios, min_cells, max_cells, cell_step,
    min_density, max_density, density_step,
    trials, base_seed=2026, output_dir="cuda_results", verbose=True, resume_csv=None,
    threads_per_block=256, num_blocks=4096, print_interval=10
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if resume_csv and Path(resume_csv).exists():
        csv_path = Path(resume_csv)
    else:
        csv_path = output_path / "experiment_data.csv"

    already_done = set()
    if resume_csv and Path(resume_csv).exists():
        with open(resume_csv, "r", encoding="utf-8") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                key = (int(row["行数"]), int(row["列数"]), float(row["地雷密度"]))
                already_done.add(key)
        if verbose:
            print(f"  续传: 已有 {len(already_done)} 个配置完成，跳过")

    ar_list = [float(x) for x in aspect_ratios]
    sizes = list(range(min_cells, max_cells + 1, cell_step))
    n_steps = int(round((max_density - min_density) / density_step)) + 1
    densities = [min_density + i * density_step for i in range(n_steps)]
    densities = [d for d in densities if d <= max_density]
    total_configs = len(ar_list) * len(sizes) * len(densities)

    all_configs = []
    for ar in ar_list:
        for total_cells in sizes:
            for density in densities:
                rows, cols = make_board_shape(total_cells, ar)
                actual_cells = rows * cols
                mines_count = max(1, min(actual_cells - 1, int(round(actual_cells * density))))
                actual_density = mines_count / actual_cells
                cfg_key = (rows, cols, round(actual_density, 4))
                if cfg_key in already_done:
                    continue
                all_configs.append({
                    "rows": rows, "cols": cols, "total_cells": actual_cells,
                    "mines": mines_count, "density": actual_density, "ar": ar,
                    "trials": trials,
                })

    if verbose:
        print("=" * 60)
        print("  CUDA 扫雷 GPU 规模效应实验（超级合并版）")
        print("=" * 60)
        print(f"  横纵比: {ar_list}")
        print(f"  棋盘规模: {min_cells} ~ {max_cells} 格, 步长 {cell_step}")
        print(f"  地雷密度: {min_density:.2f} ~ {max_density:.2f}, 步长 {density_step}")
        print(f"  试验次数: {trials} 次/配置")
        print(f"  总配置数: {total_configs} (待运行: {len(all_configs)})")
        print(f"  超级合并: 所有配置 -> 1次kernel launch")
        print("=" * 60)

    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "行数", "列数", "总格数", "地雷数", "地雷密度",
            "试验次数", "获胜次数", "胜率", "胜率标准误差", "胜率95%CI",
            "平均步数", "步数标准差", "步数95%CI",
            "平均标旗数", "标旗标准差", "标旗95%CI",
            "横纵比",
        ])
    csv_file.flush()

    # 计算实际需要的 max_cells（根据所有配置中的最大棋盘）
    actual_max_cells = 256  # 默认小棋盘阈值
    if all_configs:
        actual_max_cells = max(cfg["rows"] * cfg["cols"] for cfg in all_configs)
    
    # 计算需要的总线程数
    if all_configs:
        total_games_needed = sum(cfg["trials"] for cfg in all_configs)
    else:
        total_games_needed = trials
    
    # 分配足够的线程，但不超过 num_blocks * threads_per_block
    needed_threads = min(total_games_needed, threads_per_block * num_blocks)
    needed_threads = max(needed_threads, threads_per_block)  # 至少一个 block
    
    # 根据实际需要的线程数分配显存池
    pool_threads = needed_threads * 2  # 乘以 2 是为了留余量
    gpu_pool = GPUMemoryPool(pool_threads, actual_max_cells)
    
    if verbose:
        used, total = get_vram_usage_mb()
        print(f"  显存池: {pool_threads} threads x {actual_max_cells} cells")
        print(f"  显存使用: {used:.0f} MB / {total:.0f} MB")

    points = []
    overall_start = time.perf_counter()

    if all_configs:
        batch_start = time.perf_counter()
        results = run_super_merged_all(all_configs, base_seed, gpu_pool)
        batch_elapsed = time.perf_counter() - batch_start

        for cfg, result in zip(all_configs, results):
            wins = result["won"]
            success_rate = result["success_rate"]
            avg_steps = result["avg_steps"]
            std_steps = result["std_steps"]
            steps_ci95 = result["steps_ci95"]
            avg_flags = result["avg_flags"]
            std_flags = result["std_flags"]
            flags_ci95 = result["flags_ci95"]
            se_p = result["success_rate_se"]
            ci95_p = result["success_rate_ci95"]

            point = {
                "rows": cfg["rows"], "cols": cfg["cols"], "total_cells": cfg["total_cells"],
                "mines": cfg["mines"], "mine_density": cfg["density"],
                "trials": trials, "wins": wins, "success_rate": success_rate,
                "success_rate_se": se_p, "success_rate_ci95": ci95_p,
                "avg_steps": avg_steps, "std_steps": std_steps, "steps_ci95": steps_ci95,
                "avg_flags": avg_flags, "std_flags": std_flags, "flags_ci95": flags_ci95,
                "aspect_ratio": cfg["ar"],
            }
            points.append(point)
            writer.writerow([
                cfg["rows"], cfg["cols"], cfg["total_cells"], cfg["mines"], f"{cfg['density']:.4f}",
                trials, wins,
                f"{success_rate:.6f}", f"{se_p:.6f}", f"{ci95_p:.6f}",
                f"{avg_steps:.2f}", f"{std_steps:.2f}", f"{steps_ci95:.2f}",
                f"{avg_flags:.2f}", f"{std_flags:.2f}", f"{flags_ci95:.2f}",
                f"{cfg['ar']:.2f}",
            ])
            csv_file.flush()

            pct = (len(points) + len(already_done)) / total_configs * 100
            sys.stdout.write(f"PROGRESS|pct={pct:.2f}|done_cfgs={len(points) + len(already_done)}|total_cfgs={total_configs}|done_games={trials}|total_games={trials}|ar={cfg['ar']:.2f}|cells={cfg['total_cells']}|density={cfg['density']:.4f}\n")
            sys.stdout.flush()

        if verbose:
            elapsed = time.perf_counter() - overall_start
            print(
                f"[{len(points) + len(already_done)}/{total_configs}] 全部完成, "
                f"耗时 {batch_elapsed:.1f}s, "
                f"速度={len(all_configs) * trials / batch_elapsed:.0f} 局/秒  "
                f"(GPU时间 {elapsed:.1f}s)",
                flush=True,
            )

    csv_file.close()
    gpu_pool.free()
    overall_elapsed = time.perf_counter() - overall_start
    if verbose:
        print("\n" + "=" * 60)
        print("  实验完成!")
        print(f"  数据文件: {csv_path.resolve()}")
        print(f"  总耗时: {overall_elapsed:.1f}s")
        total_done = len(points) + len(already_done)
        print(f"  平均速度: {total_done * trials / overall_elapsed:.0f} 局/秒")
        print("=" * 60)

    return points

def parse_args():
    parser = argparse.ArgumentParser(
        description="CUDA 扫雷 GPU 规模效应实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--aspect-ratios", type=str, default="1.0", help="横纵比列表（逗号分隔）")
    parser.add_argument("--min-cells", type=int, default=16, help="最小棋盘总格数")
    parser.add_argument("--max-cells", type=int, default=256, help="最大棋盘总格数")
    parser.add_argument("--cell-step", type=int, default=24, help="棋盘格数步长")
    parser.add_argument("--min-density", type=float, default=0.10, help="最小地雷密度")
    parser.add_argument("--max-density", type=float, default=0.35, help="最大地雷密度")
    parser.add_argument("--density-step", type=float, default=0.05, help="地雷密度步长")
    parser.add_argument("--trials", type=int, default=10000, help="每个配置的重复试验次数（优化后默认10000）")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子偏移")
    parser.add_argument("--output-dir", type=str, default="cuda_results", help="输出目录")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    parser.add_argument("--verbose", action="store_true", help="详细打印（每条配置都打印）")
    parser.add_argument("--timestamp", action="store_true", help="添加时间戳到输出目录名")
    parser.add_argument("--no-interactive", action="store_true", help="不生成 HTML 交互式图表")
    parser.add_argument("--vram-monitor", action="store_true", help="显存监控")
    parser.add_argument("--resume", type=str, help="从已有 CSV 续传")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--print-interval", type=int, default=10, help="打印间隔（每 N 个配置打印一次，默认 10）")
    return parser.parse_args()

def get_vram_usage_mb():
    try:
        cudart = get_cudart()
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        err = cudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if err == 0:
            return (total.value - free.value) / (1024 * 1024), total.value / (1024 * 1024)
    except Exception:
        pass
    return 0, 0

def main():
    args = parse_args()
    aspect_ratios = [float(x) for x in args.aspect_ratios.split(",")]
    output_dir = args.output_dir
    if args.timestamp:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir + f"_{ts}"
    if args.vram_monitor:
        vram_used, vram_total = get_vram_usage_mb()
        print(f"  显存: {vram_used:.0f} MB / {vram_total:.0f} MB")
    print_interval = args.print_interval if args.verbose else 10
    run_full_experiments(
        aspect_ratios=aspect_ratios, min_cells=args.min_cells, max_cells=args.max_cells,
        cell_step=args.cell_step, min_density=args.min_density, max_density=args.max_density,
        density_step=args.density_step, trials=args.trials, base_seed=args.seed,
        output_dir=output_dir, verbose=not args.quiet, resume_csv=args.resume,
        print_interval=print_interval,
    )

if __name__ == "__main__":
    main()
