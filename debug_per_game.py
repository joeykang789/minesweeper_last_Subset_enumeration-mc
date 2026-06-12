import sys
sys.path.insert(0, '.')
from philox_python import PhiloxRNG
from run_cuda_experiments import GPUMemoryPool, run_super_merged_all
import numpy as np

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

        if not changed:
            constraints = []
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
    total = rows * cols
    safe, mines = infer_safe_mines(rows, cols, revealed, flagged, adj_counts)
    for m in mines:
        if not flagged[m]:
            flagged[m] = 1
            return m, 0
    for s in safe:
        if not revealed[s]:
            return s, 1
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
    rows, cols, mines = 8, 8, 10
    trials = 100

    print(f"对比 {trials} 局逐局结果 (seed 2026~{2026+trials-1})")
    print("=" * 60)

    py_results = []
    for t in range(trials):
        won, steps = run_py_sim(rows, cols, mines, 2026 + t)
        py_results.append((won, steps))

    gpu_pool = GPUMemoryPool(trials * 2, rows * cols)
    cfgs = [{
        "rows": rows, "cols": cols, "total_cells": rows * cols,
        "mines": mines, "density": mines / (rows * cols), "ar": 1.0,
        "trials": trials
    }]
    results = run_super_merged_all(cfgs, 2026, gpu_pool)

    import ctypes
    from run_cuda_experiments import get_cuda_lib, get_cudart, cuda_malloc, cuda_free, cuda_device_synchronize
    cudart = get_cudart()

    total_games = trials
    h_win = np.zeros(total_games, dtype=np.int32)
    h_steps = np.zeros(total_games, dtype=np.int32)
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

    match = 0
    mismatch = 0
    py_win_gpu_lose = 0
    py_lose_gpu_win = 0
    both_win = 0
    both_lose = 0

    for t in range(trials):
        py_won, py_s = py_results[t]
        gpu_won = int(h_win[t])
        gpu_s = int(h_steps[t])
        if py_won == gpu_won:
            match += 1
            if py_won:
                both_win += 1
            else:
                both_lose += 1
        else:
            mismatch += 1
            if py_won and not gpu_won:
                py_win_gpu_lose += 1
            else:
                py_lose_gpu_win += 1
            if mismatch <= 20:
                print(f"  seed={2026+t}: PY={'WIN' if py_won else 'LOSE'}(steps={py_s}) vs GPU={'WIN' if gpu_won else 'LOSE'}(steps={gpu_s})")

    print(f"\n总结:")
    print(f"  匹配: {match}/{trials} (双方赢: {both_win}, 双方输: {both_lose})")
    print(f"  不匹配: {mismatch}/{trials}")
    print(f"    PY赢GPU输: {py_win_gpu_lose}")
    print(f"    PY输GPU赢: {py_lose_gpu_win}")
    print(f"  PY胜率: {sum(1 for w,_ in py_results if w)}/{trials}")
    print(f"  GPU胜率: {int(h_win.sum())}/{trials}")


if __name__ == "__main__":
    main()
