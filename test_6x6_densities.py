"""
6×6 棋盘，5000 局，地雷密度 0.1~0.4
CPU (Python模拟) vs GPU 对比
"""
import sys, time
import numpy as np
sys.path.insert(0, '.')
from philox_python import PhiloxRNG
from run_cuda_experiments import GPUMemoryPool, run_super_merged_all

DR = [-1, -1, -1, 0, 0, 1, 1, 1]
DC = [-1, 0, 1, -1, 1, -1, 0, 1]


def cpu_place_mines(rows, cols, mines, seed):
    total = rows * cols
    first_r = seed % rows
    first_c = (seed // rows) % cols
    first_idx = first_r * cols + first_c
    cells = [i for i in range(total) if i != first_idx]
    rng = PhiloxRNG(seed=seed, counter=0)
    rng.shuffle(cells, base_counter=0)
    mine_cells = [0] * total
    for i in range(mines):
        mine_cells[cells[i]] = 1
    return mine_cells, first_r, first_c


def compute_adj_counts(rows, cols, mine_cells):
    adj = [0] * (rows * cols)
    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        count = 0
        for i in range(8):
            nr, nc = r + DR[i], c + DC[i]
            if 0 <= nr < rows and 0 <= nc < cols:
                count += mine_cells[nr * cols + nc]
        adj[idx] = count
    return adj


def infer_safe_mines(rows, cols, revealed, flagged, adj_counts):
    total = rows * cols
    seen = [0] * ((total + 31) // 32)
    safe, mines = [], []
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if not revealed[idx]:
                    continue
                n = adj_counts[idx]
                fc, hc = 0, 0
                hc_list = []
                for i in range(8):
                    nr, nc = r + DR[i], c + DC[i]
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    ni = nr * cols + nc
                    if flagged[ni]:
                        fc += 1
                    elif not revealed[ni]:
                        hc_list.append(ni)
                        hc += 1
                rem = n - fc
                if rem == 0 and hc > 0:
                    for ni in hc_list:
                        w, b = ni >> 5, ni & 31
                        if not (seen[w] & (1 << b)):
                            seen[w] |= (1 << b)
                            safe.append(ni)
                            changed = True
                elif rem > 0 and rem == hc:
                    for ni in hc_list:
                        w, b = ni >> 5, ni & 31
                        if not (seen[w] & (1 << b)):
                            seen[w] |= (1 << b)
                            mines.append(ni)
                            changed = True

        if not changed:
            constraints = []
            for r in range(rows):
                for c in range(cols):
                    idx = r * cols + c
                    if not revealed[idx]:
                        continue
                    n = adj_counts[idx]
                    fc, hc = 0, 0
                    hc_list = []
                    for i in range(8):
                        nr, nc = r + DR[i], c + DC[i]
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        ni = nr * cols + nc
                        if flagged[ni]:
                            fc += 1
                        elif not revealed[ni]:
                            hc_list.append(ni)
                            hc += 1
                    rem = n - fc
                    if hc > 0:
                        constraints.append((set(hc_list), hc, rem))

            for i in range(len(constraints)):
                if changed:
                    break
                si, hc_i, rem_i = constraints[i]
                for j in range(len(constraints)):
                    if changed:
                        break
                    if i == j:
                        continue
                    sj, hc_j, rem_j = constraints[j]
                    if hc_j > hc_i:
                        continue
                    if not sj.issubset(si):
                        continue
                    diff = si - sj
                    diff_rem = rem_i - rem_j
                    diff_sz = len(diff)
                    if diff_sz == 0 or diff_rem < 0 or diff_rem > diff_sz:
                        continue
                    if diff_rem == 0:
                        for cell in diff:
                            w, b = cell >> 5, cell & 31
                            if not (seen[w] & (1 << b)):
                                seen[w] |= (1 << b)
                                safe.append(cell)
                                changed = True
                    elif diff_rem == diff_sz:
                        for cell in diff:
                            w, b = cell >> 5, cell & 31
                            if not (seen[w] & (1 << b)):
                                seen[w] |= (1 << b)
                                mines.append(cell)
                                changed = True
    return safe, mines


def ai_choose_sh_py(rows, cols, revealed, flagged, adj_counts):
    safe, mines = infer_safe_mines(rows, cols, revealed, flagged, adj_counts)
    for mi in mines:
        if not flagged[mi]:
            return mi, 0
    for si in safe:
        if not revealed[si]:
            return si, 1

    total = rows * cols
    seen = [0] * ((total + 31) // 32)
    candidates = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if not revealed[idx]:
                continue
            for i in range(8):
                nr, nc = r + DR[i], c + DC[i]
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                ni = nr * cols + nc
                if revealed[ni] or flagged[ni]:
                    continue
                w, b = ni >> 5, ni & 31
                if not (seen[w] & (1 << b)):
                    seen[w] |= (1 << b)
                    candidates.append(ni)

    best, min_p = -1, 1.0
    for ci in candidates:
        r, c = ci // cols, ci % cols
        sum_p, cnt = 0.0, 0.0
        for k in range(8):
            nr, nc = r + DR[k], c + DC[k]
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            ni = nr * cols + nc
            if not revealed[ni]:
                continue
            n = adj_counts[ni]
            fc = hc = 0
            for j in range(8):
                nnr, nnc = nr + DR[j], nc + DC[j]
                if not (0 <= nnr < rows and 0 <= nnc < cols):
                    continue
                nni = nnr * cols + nnc
                if flagged[nni]:
                    fc += 1
                elif not revealed[nni]:
                    hc += 1
            if hc > 0:
                sum_p += max(0, n - fc) / hc
                cnt += 1.0
        p = min(1.0, sum_p / cnt) if cnt > 0 else 0.5
        if p < min_p:
            min_p, best = p, ci
    return best, 1


def flood_fill(rows, cols, r, c, mine_cells, revealed, flagged):
    q = [(r, c)]
    head = 0
    hit, revealed_cnt = False, 0
    while head < len(q):
        cr, cc = q[head]
        head += 1
        idx = cr * cols + cc
        if revealed[idx] or flagged[idx]:
            continue
        revealed[idx] = 1
        revealed_cnt += 1
        if mine_cells[idx]:
            hit = True
            break
        adj = sum(mine_cells[(cr+dr)*cols + (cc+dc)] for dr, dc in zip(DR, DC)
                  if 0 <= cr+dr < rows and 0 <= cc+dc < cols)
        if adj == 0:
            for i in range(8):
                nr, nc = cr + DR[i], cc + DC[i]
                if 0 <= nr < rows and 0 <= nc < cols:
                    ni = nr * cols + nc
                    if not revealed[ni] and not flagged[ni]:
                        q.append((nr, nc))
    return hit, revealed_cnt


def run_py_sim(rows, cols, mines, seed):
    total = rows * cols
    mine_cells, first_r, first_c = cpu_place_mines(rows, cols, mines, seed)
    adj_counts = compute_adj_counts(rows, cols, mine_cells)
    revealed = [0] * total
    flagged = [0] * total
    safe_remaining = total - mines

    hit, rc = flood_fill(rows, cols, first_r, first_c, mine_cells, revealed, flagged)
    if hit:
        return False, 1
    safe_remaining -= rc
    if safe_remaining == 0:
        return True, 1

    steps = 1
    for _ in range(total * 5):
        target, action = ai_choose_sh_py(rows, cols, revealed, flagged, adj_counts)
        if target < 0:
            break
        if action == 0:
            flagged[target] = 1
        else:
            hit, rc = flood_fill(rows, cols, target // cols, target % cols, mine_cells, revealed, flagged)
            if hit:
                return False, steps + 1
            safe_remaining -= rc
            steps += 1
            if safe_remaining == 0:
                return True, steps
    return safe_remaining == 0, steps


def main():
    rows, cols = 6, 6
    total = rows * cols
    trials = 5000
    densities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    print("=" * 70)
    print(f"  6×6 棋盘, {trials} 局, 地雷密度 0.1~0.4")
    print("=" * 70)

    configs = []
    for d in densities:
        mines = max(1, round(total * d))
        configs.append({"density": d, "mines": mines})

    # GPU
    print("\n--- GPU 测试 ---")
    gpu_configs = []
    for cfg in configs:
        gpu_configs.append({
            "rows": rows, "cols": cols, "total_cells": total,
            "mines": cfg["mines"], "density": cfg["density"], "ar": 1.0,
            "trials": trials,
        })
    max_cells = total
    total_games = trials * len(configs)
    gpu_pool = GPUMemoryPool(total_games * 2, max_cells)
    t0 = time.perf_counter()
    results_gpu = run_super_merged_all(gpu_configs, 2026, gpu_pool)
    gpu_time = time.perf_counter() - t0

    # CPU
    print("\n--- CPU (Python) 测试 ---")
    cpu_results = []
    t0 = time.perf_counter()
    for cfg in configs:
        wins = 0
        steps_list = []
        for t in range(trials):
            won, steps = run_py_sim(rows, cols, cfg["mines"], 2026 + t)
            if won:
                wins += 1
            steps_list.append(steps)
        steps_arr = np.array(steps_list, dtype=np.float64)
        p = wins / trials
        se_p = np.sqrt(p * (1 - p) / trials)
        mean_s = float(steps_arr.mean())
        std_s = float(steps_arr.std(ddof=1))
        se_s = std_s / np.sqrt(trials)
        cpu_results.append({
            "wins": wins, "success_rate": p, "success_rate_se": se_p, "success_rate_ci95": 1.96 * se_p,
            "avg_steps": mean_s, "std_steps": std_s, "steps_ci95": 1.96 * se_s,
        })
    cpu_time = time.perf_counter() - t0

    # 对比
    print("\n" + "=" * 90)
    print(f"  6×6 棋盘, {trials} 局, 高斯分布统计 (均值 ± 95%CI)")
    print("=" * 90)
    hdr = (f"{'密度':>6} {'雷':>3} | "
           f"{'CPU胜率':>14} {'GPU胜率':>14} | "
           f"{'CPU步数':>16} {'GPU步数':>16}")
    print(hdr)
    print("-" * 90)
    for i, cfg in enumerate(configs):
        c = cpu_results[i]
        g = results_gpu[i]
        print(f"{cfg['density']:>6.2f} {cfg['mines']:>3} | "
              f"{c['success_rate']:.4f}±{c['success_rate_ci95']:.4f} "
              f"{g['success_rate']:.4f}±{g['success_rate_ci95']:.4f} | "
              f"{c['avg_steps']:.2f}±{c['steps_ci95']:.2f}  "
              f"{g['avg_steps']:.2f}±{g['steps_ci95']:.2f}")
    print("-" * 90)
    print(f"GPU 总耗时: {gpu_time:.2f}s | CPU 总耗时: {cpu_time:.2f}s")


if __name__ == "__main__":
    main()
