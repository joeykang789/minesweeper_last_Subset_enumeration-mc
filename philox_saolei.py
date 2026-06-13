"""
使用 Philox RNG 的扫雷环境
与 CUDA 版本完全兼容
"""
from __future__ import annotations

import math
import random as _random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from philox_python import PhiloxRNG

Action = Tuple[int, int]
State = np.ndarray


class PhiloxMinesweeperEnv:
    """
    使用 Philox RNG 的扫雷环境
    与 CUDA 版本的 minesweeper_game.h 完全兼容
    """
    def __init__(self, rows: int = 8, cols: int = 8, mines: int = 10, seed: int | None = None):
        if mines >= rows * cols:
            raise ValueError("地雷数量必须小于总格子数")
        self.rows = rows
        self.cols = cols
        self.mines = mines
        # Philox RNG，使用与 CUDA 相同的 seed 语义
        self.rng = PhiloxRNG(seed=seed, counter=0)
        self.reset()

    def reset(self) -> State:
        # 与 CUDA place_mines_sh 一致：排除第一步位置
        total = self.rows * self.cols
        first_r = self.rng.seed % self.rows
        first_c = (self.rng.seed // self.rows) % self.cols
        first_idx = first_r * self.cols + first_c

        # 构建所有格子（排除第一个格子）
        all_cells = [i for i in range(total) if i != first_idx]
        self.rng.shuffle(all_cells)
        # 存储为 (r, c) 元组集合，与 CUDA 的 mine_cells[] 数组对应
        self.mine_cells = set()
        for i in range(self.mines):
            r = all_cells[i] // self.cols
            c = all_cells[i] % self.cols
            self.mine_cells.add((r, c))

        self.revealed = [[False] * self.cols for _ in range(self.rows)]
        self.flagged = [[False] * self.cols for _ in range(self.rows)]
        self.done = False
        self.won = False
        self.first_move = True
        return self.state()

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r: int, c: int) -> Iterable[Action]:
        for rr in range(max(0, r - 1), min(self.rows, r + 2)):
            for cc in range(max(0, c - 1), min(self.cols, c + 2)):
                if (rr, cc) != (r, c):
                    yield rr, cc

    def adjacent_mines(self, r: int, c: int) -> int:
        return sum((nr, nc) in self.mine_cells for nr, nc in self.neighbors(r, c))

    def legal_actions(self) -> List[Action]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if not self.revealed[r][c]]

    def _all_safe_revealed(self) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.mine_cells and not self.revealed[r][c]:
                    return False
        return True

    def reveal(self, r: int, c: int) -> None:
        if self.done or not self.in_bounds(r, c) or self.revealed[r][c] or self.flagged[r][c]:
            return
        if self.first_move and (r, c) in self.mine_cells:
            self._relocate_mine((r, c))
        self.first_move = False
        if (r, c) in self.mine_cells:
            self.revealed[r][c] = True
            self.done = True
            self.won = False
            return

        q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            if self.revealed[cr][cc] or self.flagged[cr][cc]:
                continue
            self.revealed[cr][cc] = True
            if self.adjacent_mines(cr, cc) == 0:
                for nr, nc in self.neighbors(cr, cc):
                    if not self.revealed[nr][nc] and (nr, nc) not in self.mine_cells:
                        q.append((nr, nc))

        if self._all_safe_revealed():
            self.done = True
            self.won = True

    def toggle_flag(self, r: int, c: int) -> None:
        if self.done or not self.in_bounds(r, c) or self.revealed[r][c]:
            return
        self.flagged[r][c] = not self.flagged[r][c]

    def _relocate_mine(self, protected: Action) -> None:
        self.mine_cells.remove(protected)
        available = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if (r, c) not in self.mine_cells and (r, c) != protected]
        self.mine_cells.add(self.rng.choice(available))

    def state(self) -> State:
        revealed = np.zeros((self.rows, self.cols), dtype=np.float32)
        counts = np.zeros((self.rows, self.cols), dtype=np.float32)
        flagged = np.zeros((self.rows, self.cols), dtype=np.float32)
        hidden = np.ones((self.rows, self.cols), dtype=np.float32)
        hint = np.zeros((self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r][c]:
                    revealed[r, c] = 1.0
                    counts[r, c] = self.adjacent_mines(r, c) / 8.0
                    hidden[r, c] = 0.0
                elif self.flagged[r][c]:
                    flagged[r, c] = 1.0
                else:
                    for nr, nc in self.neighbors(r, c):
                        if self.revealed[nr][nc]:
                            hint[r, c] = 1.0
        return np.stack([revealed, counts, flagged, hidden, hint], axis=0)


class PhiloxHeuristicEngine:
    """
    使用 Philox RNG 的启发式引擎
    与 CUDA 版本的 ai_choose_sh 逻辑兼容
    """
    def __init__(self, env: PhiloxMinesweeperEnv):
        self.env = env

    def infer_safe_and_mines(self) -> Tuple[set[Action], set[Action], List[str]]:
        """
        迭代推理 - 与 CUDA 迭代版本兼容
        持续扫描直到没有新发现
        """
        safe: set[Action] = set()
        mines: set[Action] = set()
        reasons: List[str] = []
        changed = True

        while changed:
            changed = False
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    if not self.env.revealed[r][c]:
                        continue
                    n = self.env.adjacent_mines(r, c)
                    neigh = list(self.env.neighbors(r, c))
                    hidden = [(nr, nc) for nr, nc in neigh
                              if not self.env.revealed[nr][nc] and not self.env.flagged[nr][nc]]
                    flagged = sum(self.env.flagged[nr][nc] for nr, nc in neigh)
                    remaining = n - flagged
                    if remaining == 0 and hidden:
                        for cell in hidden:
                            if cell not in safe:
                                safe.add(cell)
                                reasons.append(f"{r},{c} 周围雷数已满足，{cell} 可安全翻开")
                                changed = True
                    elif remaining > 0 and remaining == len(hidden):
                        for cell in hidden:
                            if cell not in mines:
                                mines.add(cell)
                                reasons.append(f"{r},{c} 周围剩余雷数等于隐藏格数，{cell} 必为雷")
                                changed = True
        return safe, mines, reasons

    def estimate_probability(self, cell: Action) -> float:
        """
        估计指定格子的地雷概率
        与 CUDA 版本兼容：取邻居概率的平均值
        """
        r, c = cell
        total_sum = 0.0
        cnt = 0.0

        drs = [-1, -1, -1, 0, 0, 1, 1, 1]
        dcs = [-1, 0, 1, -1, 1, -1, 0, 1]

        for k in range(8):
            nr, nc = r + drs[k], c + dcs[k]
            if not self.env.in_bounds(nr, nc):
                continue
            if not self.env.revealed[nr][nc]:
                continue

            n = self.env.adjacent_mines(nr, nc)
            hc = 0
            fc = 0
            for j in range(8):
                nnr, nnc = nr + drs[j], nc + dcs[j]
                if not self.env.in_bounds(nnr, nnc):
                    continue
                nni = nnr * self.env.cols + nnc
                if self.env.flagged[nnr][nnc]:
                    fc += 1
                elif not self.env.revealed[nnr][nnc]:
                    hc += 1

            if hc > 0:
                prob = max(0.0, (n - fc) / hc)
                total_sum += prob
                cnt += 1.0

        if cnt > 0:
            return max(0.0, min(1.0, total_sum / cnt))
        return 0.5

    def monte_carlo_choose(self) -> Optional[Tuple[str, Action, str]]:
        env = self.env
        total = env.rows * env.cols
        unknown = [(r, c) for r in range(env.rows) for c in range(env.cols)
                   if not env.revealed[r][c] and not env.flagged[r][c]]
        if not unknown:
            return None
        flagged_count = sum(env.flagged[r][c] for r in range(env.rows) for c in range(env.cols))
        remaining = env.mines - flagged_count
        if remaining <= 0:
            return None

        drs = [-1, -1, -1, 0, 0, 1, 1, 1]
        dcs = [-1, 0, 1, -1, 1, -1, 0, 1]

        constraints = []
        for r in range(env.rows):
            for c in range(env.cols):
                if not env.revealed[r][c]:
                    continue
                hlist, fc = [], 0
                for k in range(8):
                    nr, nc = r + drs[k], c + dcs[k]
                    if not env.in_bounds(nr, nc):
                        continue
                    if env.flagged[nr][nc]:
                        fc += 1
                    elif not env.revealed[nr][nc]:
                        hlist.append((nr, nc))
                if not hlist:
                    continue
                rem = env.adjacent_mines(r, c) - fc
                if rem < 0 or rem > len(hlist):
                    continue
                constraints.append((hlist, rem))

        num_samples = 200
        valid_count = 0
        hit = {cell: 0 for cell in unknown}
        to_select = min(remaining, len(unknown))

        for _ in range(num_samples):
            perm = unknown[:]
            for i in range(to_select):
                j = i + _random.randrange(len(unknown) - i)
                perm[i], perm[j] = perm[j], perm[i]
            mine_set = set(perm[:to_select])

            valid = True
            for hlist, rem in constraints:
                cnt = sum(1 for cell in hlist if cell in mine_set)
                if cnt != rem:
                    valid = False
                    break

            if valid:
                valid_count += 1
                for cell in perm[:to_select]:
                    hit[cell] += 1

        if valid_count == 0:
            return None

        best, min_p = None, 1.0
        for cell in unknown:
            p = hit[cell] / valid_count
            if p < min_p:
                min_p = p
                best = cell

        if best is None:
            return None
        return "reveal", best, f"蒙特卡洛: {best} 概率 {min_p:.3f} ({valid_count}/{num_samples} 有效样本)"

    def mcmc_choose(self) -> Optional[Tuple[str, Action, str]]:
        """MCMC: Markov Chain Monte Carlo with swap-based perturbation"""
        env = self.env
        total = env.rows * env.cols
        unknown = [(r, c) for r in range(env.rows) for c in range(env.cols)
                   if not env.revealed[r][c] and not env.flagged[r][c]]
        if not unknown:
            return None
        flagged_count = sum(env.flagged[r][c] for r in range(env.rows) for c in range(env.cols))
        remaining = env.mines - flagged_count
        if remaining <= 0:
            return None
        to_select = min(remaining, len(unknown))

        drs = [-1, -1, -1, 0, 0, 1, 1, 1]
        dcs = [-1, 0, 1, -1, 1, -1, 0, 1]

        # Build constraints
        constraints = []  # each is (list_of_cells, rem)
        for r in range(env.rows):
            for c in range(env.cols):
                if not env.revealed[r][c]:
                    continue
                hlist, fc = [], 0
                for k in range(8):
                    nr, nc = r + drs[k], c + dcs[k]
                    if not env.in_bounds(nr, nc):
                        continue
                    if env.flagged[nr][nc]:
                        fc += 1
                    elif not env.revealed[nr][nc]:
                        hlist.append((nr, nc))
                if not hlist:
                    continue
                rem = env.adjacent_mines(r, c) - fc
                if rem < 0 or rem > len(hlist):
                    continue
                constraints.append((hlist, rem))

        # Cell-to-constraint mapping
        cell_cons = {cell: [] for cell in unknown}
        for ci, (hlist, _) in enumerate(constraints):
            for cell in hlist:
                cell_cons[cell].append(ci)

        # Find initial valid config via rejection sampling (up to 200 attempts)
        is_mine = {cell: False for cell in unknown}
        found = False
        for _ in range(200):
            perm = unknown[:]
            for i in range(to_select):
                j = i + _random.randrange(len(unknown) - i)
                perm[i], perm[j] = perm[j], perm[i]
            mine_set = set(perm[:to_select])
            valid = True
            for hlist, rem in constraints:
                cnt = sum(1 for cell in hlist if cell in mine_set)
                if cnt != rem:
                    valid = False
                    break
            if valid:
                for cell in mine_set:
                    is_mine[cell] = True
                found = True
                break
        if not found:
            return None

        mine_list = [c for c in unknown if is_mine[c]]
        free_list = [c for c in unknown if not is_mine[c]]
        num_mines = len(mine_list)

        # Burn-in: 100 swap perturbations (no recording)
        for _ in range(100):
            mi = _random.randrange(num_mines)
            fi = _random.randrange(len(free_list))
            mc, fc_cell = mine_list[mi], free_list[fi]

            is_mine[mc] = False
            is_mine[fc_cell] = True

            affected = set(cell_cons[mc]) | set(cell_cons[fc_cell])
            valid = True
            for ci in affected:
                hlist, rem = constraints[ci]
                cnt = sum(1 for cell in hlist if is_mine[cell])
                if cnt != rem:
                    valid = False
                    break

            if valid:
                mine_list[mi] = fc_cell
                free_list[fi] = mc
            else:
                is_mine[mc] = True
                is_mine[fc_cell] = False

        # Sampling: 200 swap perturbations (record mine counts)
        hit = {cell: 0 for cell in unknown}
        for _ in range(200):
            mi = _random.randrange(num_mines)
            fi = _random.randrange(len(free_list))
            mc, fc_cell = mine_list[mi], free_list[fi]

            is_mine[mc] = False
            is_mine[fc_cell] = True

            affected = set(cell_cons[mc]) | set(cell_cons[fc_cell])
            valid = True
            for ci in affected:
                hlist, rem = constraints[ci]
                cnt = sum(1 for cell in hlist if is_mine[cell])
                if cnt != rem:
                    valid = False
                    break

            if valid:
                mine_list[mi] = fc_cell
                free_list[fi] = mc
            else:
                is_mine[mc] = True
                is_mine[fc_cell] = False

            for cell in mine_list:
                hit[cell] += 1

        best, min_p = None, 1.0
        for cell in unknown:
            p = hit[cell] / 200.0
            if p < min_p:
                min_p, best = p, cell

        if best is None:
            return None
        return "reveal", best, f"MCMC: {best} 概率 {min_p:.3f} ({200} MCMC 样本)"

    def choose(self) -> Tuple[str, Action, str]:
        """
        选择下一步行动
        与 CUDA 版本的 ai_choose_sh 兼容
        """
        safe, mines, reasons = self.infer_safe_and_mines()

        for cell in mines:
            if not self.env.flagged[cell[0]][cell[1]]:
                return "flag", cell, reasons[-1] if reasons else f"{cell} 被推断为雷"

        for cell in safe:
            if not self.env.revealed[cell[0]][cell[1]]:
                return "reveal", cell, reasons[-1] if reasons else f"{cell} 被推断为安全"

        mc_result = self.monte_carlo_choose()
        if mc_result is not None:
            return mc_result

        candidates = []
        seen = set()
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if not self.env.revealed[r][c]:
                    continue
                for nr, nc in self.env.neighbors(r, c):
                    if (nr, nc) in seen:
                        continue
                    if not self.env.revealed[nr][nc] and not self.env.flagged[nr][nc]:
                        seen.add((nr, nc))
                        candidates.append((nr, nc))

        if not candidates:
            candidates = [cell for cell in self.env.legal_actions()
                         if not self.env.flagged[cell[0]][cell[1]]]
        if not candidates:
            return "reveal", self.env.rng.choice(self.env.legal_actions()), "没有可继续推理的未插旗格子"

        best = None
        min_prob = 1.0
        for cell in candidates:
            prob = self.estimate_probability(cell)
            if prob < min_prob:
                min_prob = prob
                best = cell

        if best is None:
            best = self.env.rng.choice(candidates)

        return "reveal", best, f"无法确定，选择雷概率最低的格子 {best}，概率约 {min_prob:.2f}"
