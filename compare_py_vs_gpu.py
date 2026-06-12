"""
直接对比 GPU 和 Python 的每一步决策
"""
import sys
sys.path.insert(0, '.')
from philox_python import PhiloxRNG
from run_cuda_experiments import GPUMemoryPool, run_super_merged_all


DR = [-1, -1, -1, 0, 0, 1, 1, 1]
DC = [-1, 0, 1, -1, 1, -1, 0, 1]


def cpu_place_mines(rows, cols, mines, seed):
    """与 CUDA 一致的地雷放置"""
    total = rows * cols
    first_r = seed % rows
    first_c = (seed // rows) % cols
    first_idx = first_r * cols + first_c

    cells = [i for i in range(total) if i != first_idx]
    rng = PhiloxRNG(seed=seed, counter=0)
    # 使用与CUDA兼容的shuffle，base_counter=0
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
    """迭代推理 - 与 CUDA infer_all_sh_iterative 完全一致"""
    total = rows * cols
    seen = [0] * ((total + 31) // 32)
    safe, mines = [], []

    changed = 1
    while changed:
        changed = 0
        for idx in range(total):
            if not revealed[idx]:
                continue
            r, c = idx // cols, idx % cols
            n = adj_counts[idx]
            fc = hc = 0
            hc_list = []

            for i in range(8):
                nr, nc = r + DR[i], c + DC[i]
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                ni = nr * cols + nc
                if flagged[ni]:
                    fc += 1
                elif not revealed[ni]:
                    hc += 1
                    hc_list.append(ni)

            rem = n - fc
            if rem == 0 and hc > 0:
                for ni in hc_list:
                    w, b = ni >> 5, ni & 31
                    if not (seen[w] & (1 << b)):
                        seen[w] |= (1 << b)
                        safe.append(ni)
                        changed = 1
            elif rem > 0 and rem == hc:
                for ni in hc_list:
                    w, b = ni >> 5, ni & 31
                    if not (seen[w] & (1 << b)):
                        seen[w] |= (1 << b)
                        mines.append(ni)
                        changed = 1
    return safe, mines


def ai_choose_sh_py(rows, cols, revealed, flagged, adj_counts):
    """与 CUDA ai_choose_sh 完全一致的选择逻辑"""
    total = rows * cols

    # 推理
    safe, mines = infer_safe_mines(rows, cols, revealed, flagged, adj_counts)

    # 标记地雷
    for m in mines:
        if not flagged[m]:
            flagged[m] = 1
            return m, 0

    # 翻开安全格
    for s in safe:
        if not revealed[s]:
            return s, 1

    # 收集边界候选
    seen = [0] * ((total + 31) // 32)
    cand = []
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
                    cand.append(ni)

    if not cand:
        return -1, -1

    # 概率估计
    best, min_p = -1, 1.0
    for ci in cand:
        r, c = ci // cols, ci % cols
        sum_p, cnt = 0.0, 0.0

        for i in range(8):
            nr, nc = r + DR[i], c + DC[i]
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

        p = cnt > 0 and min(1.0, sum_p / cnt) or 0.5
        if p < min_p:
            min_p, best = p, ci

    return best, 1


def flood_fill(rows, cols, r, c, mine_cells, revealed, flagged):
    """返回 (hit_mine, revealed_count)"""
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
    """Python 模拟"""
    total = rows * cols
    mine_cells, first_r, first_c = cpu_place_mines(rows, cols, mines, seed)
    adj_counts = compute_adj_counts(rows, cols, mine_cells)

    revealed = [0] * total
    flagged = [0] * total
    safe_remaining = total - mines

    # 第一步
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
    print("=" * 60)
    print("  对比 Python 模拟 vs GPU")
    print("=" * 60)

    # 测试 500 局
    trials = 500
    print(f"\n运行 {trials} 局 Python 模拟...")
    py_wins = 0
    for t in range(trials):
        won, _ = run_py_sim(8, 8, 10, 2026 + t)
        if won:
            py_wins += 1
    print(f"Python 胜率: {py_wins}/{trials} = {py_wins/trials:.1%}")

    # GPU 测试
    print(f"\n运行 GPU 测试...")
    gpu_pool = GPUMemoryPool(trials * 2, 64)
    cfgs = [{
        "rows": 8, "cols": 8, "total_cells": 64,
        "mines": 10, "density": 10/64, "ar": 1.0,
        "trials": trials
    }]
    results = run_super_merged_all(cfgs, 2026, gpu_pool)
    gpu_wins = results[0]["won"]
    print(f"GPU 胜率: {gpu_wins}/{trials} = {gpu_wins/trials:.1%}")

    print(f"\n差异: {abs(py_wins/trials - gpu_wins/trials):.1%}")


if __name__ == "__main__":
    main()