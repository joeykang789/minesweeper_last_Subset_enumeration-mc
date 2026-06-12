import sys, time
import numpy as np
sys.path.insert(0, '.')
from philox_python import PhiloxRNG

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


def infer_old(rows, cols, revealed, flagged, adj_counts):
    total = rows * cols
    seen = [0] * ((total + 31) // 32)
    safe, mines_list = [], []
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
                            mines_list.append(ni)
                            changed = True
    return safe, mines_list


def infer_new(rows, cols, revealed, flagged, adj_counts):
    total = rows * cols
    seen = [0] * ((total + 31) // 32)
    safe, mines_list = [], []
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
                            mines_list.append(ni)
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
                                mines_list.append(cell)
                                changed = True
    return safe, mines_list


def ai_choose(rows, cols, revealed, flagged, adj_counts, infer_func):
    safe, mines_list = infer_func(rows, cols, revealed, flagged, adj_counts)
    for mi in mines_list:
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
    hit, rc = False, 0
    while head < len(q):
        cr, cc = q[head]
        head += 1
        idx = cr * cols + cc
        if revealed[idx] or flagged[idx]:
            continue
        revealed[idx] = 1
        rc += 1
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
    return hit, rc


def run_sim(rows, cols, mines, seed, infer_func):
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
        target, action = ai_choose(rows, cols, revealed, flagged, adj_counts, infer_func)
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


def benchmark(rows, cols, mines, trials, seed_base, label):
    wins_old, wins_new = 0, 0
    steps_old_sum, steps_new_sum = 0, 0

    t0 = time.perf_counter()
    for t in range(trials):
        won, steps = run_sim(rows, cols, mines, seed_base + t, infer_old)
        if won:
            wins_old += 1
        steps_old_sum += steps
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for t in range(trials):
        won, steps = run_sim(rows, cols, mines, seed_base + t, infer_new)
        if won:
            wins_new += 1
        steps_new_sum += steps
    t_new = time.perf_counter() - t0

    return {
        'label': label,
        'mines': mines,
        'density': mines / (rows * cols),
        'trials': trials,
        'old_wr': wins_old / trials,
        'new_wr': wins_new / trials,
        'old_steps': steps_old_sum / trials,
        'new_steps': steps_new_sum / trials,
        'old_time': t_old,
        'new_time': t_new,
    }


def main():
    seed_base = 2026
    trials = 500

    configs = [
        (6, 6, 4,  "6x6 d=0.11"),
        (6, 6, 7,  "6x6 d=0.19"),
        (6, 6, 11, "6x6 d=0.31"),
        (8, 8, 10, "8x8 d=0.16"),
        (8, 8, 14, "8x8 d=0.22"),
        (10, 10, 15, "10x10 d=0.15"),
        (10, 10, 20, "10x10 d=0.20"),
        (16, 16, 40, "16x16 d=0.16"),
        (16, 16, 55, "16x16 d=0.21"),
        (20, 20, 60, "20x20 d=0.15"),
        (20, 20, 80, "20x20 d=0.20"),
    ]

    print("=" * 95)
    print("  旧算法 (单格推理) vs 新算法 (单格+子集枚举) 全面对比")
    print(f"  每个配置 {trials} 局, seed={seed_base}~{seed_base+trials-1}")
    print("=" * 95)
    hdr = (f"{'配置':<14} {'密度':>5} {'雷':>4} | "
           f"{'旧胜率':>7} {'新胜率':>7} {'提升':>7} | "
           f"{'旧步数':>7} {'新步数':>7} | "
           f"{'旧耗时':>7} {'新耗时':>7} {'加速':>6}")
    print(hdr)
    print("-" * 95)

    all_results = []
    for rows, cols, mines, label in configs:
        r = benchmark(rows, cols, mines, trials, seed_base, label)
        all_results.append(r)

        wr_diff = r['new_wr'] - r['old_wr']
        speedup = r['old_time'] / r['new_time'] if r['new_time'] > 0 else float('inf')

        wr_sign = "+" if wr_diff >= 0 else ""
        print(f"{label:<14} {r['density']:>5.2f} {r['mines']:>4} | "
              f"{r['old_wr']:>7.1%} {r['new_wr']:>7.1%} {wr_sign}{wr_diff:>6.1%} | "
              f"{r['old_steps']:>7.1f} {r['new_steps']:>7.1f} | "
              f"{r['old_time']:>6.2f}s {r['new_time']:>6.2f}s {speedup:>5.2f}x")

    print("-" * 95)

    avg_old_wr = np.mean([r['old_wr'] for r in all_results])
    avg_new_wr = np.mean([r['new_wr'] for r in all_results])
    total_old_time = sum(r['old_time'] for r in all_results)
    total_new_time = sum(r['new_time'] for r in all_results)

    print(f"{'平均/总':<14} {'':>5} {'':>4} | "
          f"{avg_old_wr:>7.1%} {avg_new_wr:>7.1%} {'+' if avg_new_wr-avg_old_wr>=0 else ''}{avg_new_wr-avg_old_wr:>6.1%} | "
          f"{'':>7} {'':>7} | "
          f"{total_old_time:>6.2f}s {total_new_time:>6.2f}s {total_old_time/total_new_time:>5.2f}x")
    print("=" * 95)

    print("\n总结:")
    wr_improvements = [r['new_wr'] - r['old_wr'] for r in all_results]
    print(f"  胜率提升: 平均 {np.mean(wr_improvements):.1%}, 最大 {max(wr_improvements):.1%}, 最小 {min(wr_improvements):.1%}")
    print(f"  总耗时: 旧 {total_old_time:.2f}s -> 新 {total_new_time:.2f}s (速度比 {total_old_time/total_new_time:.2f}x)")


if __name__ == "__main__":
    main()
