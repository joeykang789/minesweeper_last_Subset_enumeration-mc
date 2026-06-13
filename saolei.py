#!/usr/bin/env python3
"""
扫雷AI - CUDA GPU加速的蒙特卡洛扫雷求解器
统一命令行入口
"""
import sys, os, time, csv, argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

def cmd_sweep(args):
    """执行密度扫描实验"""
    from run_cuda_experiments import GPUMemoryPool, run_super_merged_all

    # 解析棋盘大小参数
    board = args.board
    if "x" in board:
        # 单个棋盘: "20x20" 或 "20"
        parts = board.split("x")
        rows = int(parts[0])
        cols = int(parts[1]) if len(parts) > 1 else rows
        board_sizes = [(rows, cols)]
    elif "-" in board:
        # 范围: "20-50" → [20,25,30,35,40,45,50]
        lo, hi = map(int, board.split("-"))
        step = getattr(args, "board_step", 5)
        board_sizes = [(s, s) for s in range(lo, hi + 1, step)]
    else:
        board_sizes = [(int(board), int(board))]

    # 解析密度参数
    rho = args.rho
    if "-" in rho:
        parts = rho.split("-")
        lo = float(parts[0])
        hi = float(parts[1])
        step = float(parts[2]) if len(parts) > 2 else 0.01
        densities = []
        d = lo
        while d <= hi + 1e-9:
            densities.append(d)
            d += step
    else:
        densities = [float(rho)]

    # 其余参数
    trials = args.trials
    seed = args.seed
    output_csv = args.output or (PROJECT_DIR / "sweep_results.csv")
    output_csv = Path(output_csv)

    # 断点重续
    completed = set()
    if output_csv.exists() and args.resume:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((int(row["rows"]), float(row["density"])))

    total_processed = 0

    for rows, cols in board_sizes:
        total = rows * cols
        configs = []
        for d in densities:
            if (rows, round(d, 4)) in completed:
                continue
            mines = max(1, min(total - 1, int(round(total * d))))
            configs.append({
                "rows": rows, "cols": cols, "total_cells": total,
                "mines": mines, "density": mines / total, "ar": 1.0,
                "trials": trials,
            })
        if not configs:
            print(f"  {rows}x{cols}: 全部完成，跳过")
            continue

        n_games = sum(c["trials"] for c in configs)
        total_processed += n_games
        print(f"\n{rows}x{cols}: {len(configs)} 密度 × {trials} 局 = {n_games} 局")

        gpu_pool = GPUMemoryPool(n_games * 2, total)
        t0 = time.perf_counter()
        results = run_super_merged_all(configs, seed, gpu_pool)
        elapsed = time.perf_counter() - t0
        gpu_pool.free()

        print(f"  耗时: {elapsed:.1f}s ({n_games/elapsed:.0f} 局/秒)")

        write_header = not output_csv.exists()
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "rows", "cols", "total_cells", "mines", "density",
                    "trials", "wins", "win_rate", "ci95", "avg_steps", "std_steps",
                ])
            for cfg, r in zip(configs, results):
                writer.writerow([
                    cfg["rows"], cfg["cols"], cfg["total_cells"],
                    cfg["mines"], f"{cfg['density']:.4f}",
                    trials, r["won"], f"{r['success_rate']:.6f}",
                    f"{r['success_rate_ci95']:.6f}",
                    f"{r['avg_steps']:.2f}", f"{r['std_steps']:.2f}",
                ])

    if total_processed == 0:
        print("没有需要运行的配置")
    else:
        print(f"\n完成！共 {total_processed} 局，结果保存到 {output_csv}")


def cmd_play(args):
    """执行单局或多局游戏"""
    from run_cuda_experiments import GPUMemoryPool, run_super_merged_all

    rows, cols = args.board, args.board
    total = rows * cols
    mines = max(1, min(total - 1, int(round(total * args.density))))
    config = {
        "rows": rows, "cols": cols, "total_cells": total,
        "mines": mines, "density": mines / total, "ar": 1.0,
        "trials": args.games,
    }

    print(f"运行 {rows}x{cols} × {args.games} 局 (地雷={mines}, ρ={mines/total:.3f})")
    gpu_pool = GPUMemoryPool(args.games * 2, total)
    t0 = time.perf_counter()
    results = run_super_merged_all([config], args.seed, gpu_pool)
    elapsed = time.perf_counter() - t0
    gpu_pool.free()

    r = results[0]
    print(f"\n结果:")
    print(f"  胜率: {r['success_rate']*100:.1f}% ± {r['success_rate_ci95']*100:.1f}%")
    print(f"  平均步数: {r['avg_steps']:.1f} ± {r['std_steps']:.1f}")
    print(f"  耗时: {elapsed:.2f}s ({args.games/elapsed:.0f} 局/秒)")


def cmd_web(args):
    """启动网页服务"""
    port = args.port
    server_script = PROJECT_DIR / "web" / "server.py"
    os.chdir(str(PROJECT_DIR))
    os.system(f'"{sys.executable}" "{server_script}" {port}')


def cmd_results(args):
    """查看结果摘要"""
    import numpy as np
    csv_path = args.csv
    if not csv_path or not os.path.exists(csv_path):
        print("请指定有效的 CSV 文件路径: --csv <文件>")
        return

    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = int(row["rows"])
            if key not in data:
                data[key] = []
            data[key].append({
                "density": float(row["density"]),
                "win_rate": float(row["win_rate"]) * 100,
                "ci95": float(row["ci95"]) * 100,
            })

    print(f"{'棋盘':>8} {'密度数':>6} {'临界ρ':>8} {'ρ50%':>8} {'ρ85%':>8}")
    print("-" * 45)
    for size in sorted(data.keys()):
        d = sorted(data[size])
        rho_crit = next((den for den, w, _ in [(x["density"], x["win_rate"], x["ci95"]) for x in d] if w < 5), 0)
        rho_50 = 0
        for i in range(1, len(d)):
            if d[i-1]["win_rate"] >= 50 and d[i]["win_rate"] < 50:
                rho_50 = (d[i-1]["density"] + d[i]["density"]) / 2
                break
        rho_85 = 0
        for i in range(1, len(d)):
            if d[i-1]["win_rate"] >= 85 and d[i]["win_rate"] < 85:
                rho_85 = (d[i-1]["density"] + d[i]["density"]) / 2
                break
        total = size * size
        print(f"{size}x{size:<3} {total:5d} {len(d):6d} {rho_crit:8.2f} {rho_50:8.2f} {rho_85:8.2f}")


def cmd_info(args):
    """显示系统信息和算法说明"""
    print("=" * 60)
    print("  扫雷AI - 子集枚举 + 蒙特卡洛(MC)求解器")
    print("=" * 60)
    print()
    print("算法栈:")
    print("  ① 单格推理 (single-cell constraint propagation)")
    print("  ② 子集枚举推理 (subset enumeration)")
    print("  ③ 蒙特卡洛拒绝采样 (Monte Carlo rejection sampling)")
    print("     - Frontier-only MC: 仅对边界格采样，内部格用均匀概率")
    print("     - 自适应采样数: 小棋盘200 → 大棋盘50")
    print("     - 对大棋盘(>400格)跳过子集枚举")
    print()
    print("性能特征:")
    print("  - 20×20 密度扫描:   ~6000+ 局/秒")
    print("  - 50×50 密度扫描:   ~100 局/秒")
    print()
    print("使用方式:")
    print("  python saolei.py sweep --board 20x20 --rho 0.10-0.35 --trials 200")
    print("  python saolei.py play --board 20 --density 0.20 --games 1000")
    print("  python saolei.py web [port]")
    print("  python saolei.py results --csv sweep_results.csv")
    print("  python saolei.py info")


def main():
    parser = argparse.ArgumentParser(
        description="扫雷AI - CUDA GPU加速的蒙特卡洛扫雷求解器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # sweep
    p_sweep = sub.add_parser("sweep", help="密度扫描实验")
    p_sweep.add_argument("--board", default="20x20", help="棋盘大小, e.g. 20x20, 20-50")
    p_sweep.add_argument("--board-step", type=int, default=5, help="棋盘步长(范围模式)")
    p_sweep.add_argument("--rho", default="0.10-0.35", help="密度范围, e.g. 0.10-0.35 or 0.10-0.35:0.01")
    p_sweep.add_argument("--trials", type=int, default=200, help="每配置局数")
    p_sweep.add_argument("--seed", type=int, default=2026, help="随机种子")
    p_sweep.add_argument("--output", type=str, default=None, help="输出CSV路径")
    p_sweep.add_argument("--resume", action="store_true", help="断点重续")

    # play
    p_play = sub.add_parser("play", help="玩游戏")
    p_play.add_argument("--board", type=int, default=20, help="棋盘大小")
    p_play.add_argument("--density", type=float, default=0.20, help="地雷密度")
    p_play.add_argument("--games", type=int, default=1000, help="局数")
    p_play.add_argument("--seed", type=int, default=2026, help="随机种子")

    # web
    p_web = sub.add_parser("web", help="启动网页服务")
    p_web.add_argument("port", type=int, nargs="?", default=8080, help="端口号")

    # results
    p_results = sub.add_parser("results", help="查看结果摘要")
    p_results.add_argument("--csv", type=str, default=None, help="CSV结果文件")

    # info
    sub.add_parser("info", help="显示系统信息")

    args = parser.parse_args()

    if args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "results":
        cmd_results(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()