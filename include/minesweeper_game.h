#ifndef MINESWEEPER_GAME_H
#define MINESWEEPER_GAME_H

#include "minesweeper_types.h"
#include "philox_rng.h"

static const __device__ int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const __device__ int dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

__device__ __forceinline__ int in_bounds(int r, int c, int rows, int cols) {
    return r >= 0 && r < rows && c >= 0 && c < cols;
}

__device__ __forceinline__ void do_reveal_sh(int r, int c, int rows, int cols,
                                                int* revealed, int* flagged,
                                                const int* mine_cells,
                                                int* q_r, int* q_c,
                                                int* safe_remaining) {
    if (revealed[r * cols + c] || flagged[r * cols + c]) return;
    int head = 0, tail = 0;
    q_r[tail] = r; q_c[tail] = c; tail++;
    while (head < tail) {
        int cr = q_r[head], cc = q_c[head]; head++;
        int idx = cr * cols + cc;
        if (revealed[idx] || flagged[idx]) continue;
        revealed[idx] = 1;
        if (!mine_cells[idx]) (*safe_remaining)--;
        int adj = 0;
        for (int i = 0; i < 8; i++) {
            int nr = cr + dr[i], nc = cc + dc[i];
            if (!in_bounds(nr, nc, rows, cols)) continue;
            adj += mine_cells[nr * cols + nc];
        }
        if (adj == 0) {
            for (int i = 0; i < 8; i++) {
                int nr = cr + dr[i], nc = cc + dc[i];
                if (in_bounds(nr, nc, rows, cols) && !revealed[nr * cols + nc] && !flagged[nr * cols + nc]) {
                    q_r[tail] = nr; q_c[tail] = nc; tail++;
                }
            }
        }
    }
}

__device__ __forceinline__ void place_mines_sh(int rows, int cols, int mines,
                                                  int first_r, int first_c,
                                                  int* mine_cells, uint32_t seed,
                                                  int* all_cells_buf) {
    int total = rows * cols;
    int cell = first_r * cols + first_c;
    for (int i = 0; i < total; i++) mine_cells[i] = 0;
    int idx = 0;
    for (int i = 0; i < total; i++) {
        if (i != cell) all_cells_buf[idx++] = i;
    }
    philox_shuffle(all_cells_buf, idx, seed, 0);
    for (int i = 0; i < mines; i++) {
        mine_cells[all_cells_buf[i]] = 1;
    }
}

__device__ __forceinline__ int infer_all_sh(int rows, int cols,
                                              const int* revealed, const int* flagged,
                                              const int* mine_cells,
                                              int* out_safe, int* out_mines,
                                              uint32_t* seen) {
    int safe_count = 0, mine_count = 0;
    int total_words = (rows * cols + 31) / 32;
    for (int w = 0; w < total_words; w++) seen[w] = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            if (!revealed[idx]) continue;

            int n = 0, fc = 0, hc = 0;
            int hc_list[8], hc_count = 0;
            
            for (int i = 0; i < 8; i++) {
                int nr = r + dr[i], nc = c + dc[i];
                if (!in_bounds(nr, nc, rows, cols)) continue;
                int ni = nr * cols + nc;
                n += mine_cells[ni];
                if (flagged[ni]) fc++;
                else if (!revealed[ni]) { hc_list[hc_count++] = ni; hc++; }
            }

            int rem = n - fc;
            if (rem == 0 && hc > 0) {
                for (int k = 0; k < hc_count; k++) {
                    int ni = hc_list[k];
                    int word = ni >> 5, bit = ni & 31;
                    if (!(seen[word] & (1u << bit))) {
                        seen[word] |= (1u << bit);
                        out_safe[safe_count++] = ni;
                    }
                }
            } else if (rem > 0 && rem == hc) {
                for (int k = 0; k < hc_count; k++) {
                    int ni = hc_list[k];
                    int word = ni >> 5, bit = ni & 31;
                    if (!(seen[word] & (1u << bit))) {
                        seen[word] |= (1u << bit);
                        out_mines[mine_count++] = ni;
                    }
                }
            }
        }
    }
    return (mine_count << 16) | (safe_count & 0xFFFF);
}

__device__ __forceinline__ void get_neighbor_info(int r, int c, int rows, int cols,
                                                    const int* revealed, const int* flagged,
                                                    const int* mine_cells,
                                                    int* out_n, int* out_fc, int* out_hc) {
    int n = 0, fc = 0, hc = 0;
    for (int i = 0; i < 8; i++) {
        int nr = r + dr[i], nc = c + dc[i];
        if (!in_bounds(nr, nc, rows, cols)) continue;
        int ni = nr * cols + nc;
        n += mine_cells[ni];
        if (flagged[ni]) fc++;
        else if (!revealed[ni]) hc++;
    }
    *out_n = n; *out_fc = fc; *out_hc = hc;
}

__device__ __forceinline__ int infer_all_sh_iterative(int rows, int cols,
                                                       const int* revealed, const int* flagged,
                                                       const int* mine_cells,
                                                       int* out_safe, int* out_mines,
                                                       uint32_t* seen) {
    int total_cells = rows * cols;
    int total_words = (total_cells + 31) / 32;
    for (int w = 0; w < total_words; w++) seen[w] = 0;
    int safe_count = 0, mine_count = 0;
    int changed = 1;

    while (changed) {
        changed = 0;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                if (!revealed[idx]) continue;

                int n = 0, fc = 0, hc = 0;
                int hc_list[8], hc_count = 0;

                for (int i = 0; i < 8; i++) {
                    int nr = r + dr[i], nc = c + dc[i];
                    if (!in_bounds(nr, nc, rows, cols)) continue;
                    int ni = nr * cols + nc;
                    n += mine_cells[ni];
                    if (flagged[ni]) fc++;
                    else if (!revealed[ni]) { hc_list[hc_count++] = ni; hc++; }
                }

                int rem = n - fc;
                if (rem == 0 && hc > 0) {
                    for (int k = 0; k < hc_count; k++) {
                        int ni = hc_list[k];
                        int word = ni >> 5, bit = ni & 31;
                        if (!(seen[word] & (1u << bit))) {
                            seen[word] |= (1u << bit);
                            out_safe[safe_count++] = ni;
                            changed = 1;
                        }
                    }
                } else if (rem > 0 && rem == hc) {
                    for (int k = 0; k < hc_count; k++) {
                        int ni = hc_list[k];
                        int word = ni >> 5, bit = ni & 31;
                        if (!(seen[word] & (1u << bit))) {
                            seen[word] |= (1u << bit);
                            out_mines[mine_count++] = ni;
                            changed = 1;
                        }
                    }
                }
            }
        }

        if (!changed) {
            int num_c = 0;
            int ch[128 * 8];
            int chc[128];
            int crm[128];

            for (int rr = 0; rr < rows && num_c < 128; rr++) {
                for (int cc = 0; cc < cols && num_c < 128; cc++) {
                    int idx = rr * cols + cc;
                    if (!revealed[idx]) continue;
                    int n = 0, fc = 0, hc = 0;
                    int hc_list[8], hc_count = 0;
                    for (int i = 0; i < 8; i++) {
                        int nr = rr + dr[i], nc = cc + dc[i];
                        if (!in_bounds(nr, nc, rows, cols)) continue;
                        int ni = nr * cols + nc;
                        n += mine_cells[ni];
                        if (flagged[ni]) fc++;
                        else if (!revealed[ni]) { hc_list[hc_count++] = ni; hc++; }
                    }
                    int rem = n - fc;
                    if (hc > 0) {
                        chc[num_c] = hc;
                        crm[num_c] = rem;
                        for (int k = 0; k < hc_count; k++)
                            ch[num_c * 8 + k] = hc_list[k];
                        num_c++;
                    }
                }
            }

            for (int i = 0; i < num_c && !changed; i++) {
                for (int j = 0; j < num_c && !changed; j++) {
                    if (i == j || chc[j] > chc[i]) continue;

                    int is_sub = 1;
                    for (int a = 0; a < chc[j] && is_sub; a++) {
                        int found = 0;
                        for (int b = 0; b < chc[i]; b++) {
                            if (ch[j * 8 + a] == ch[i * 8 + b]) { found = 1; break; }
                        }
                        if (!found) is_sub = 0;
                    }
                    if (!is_sub) continue;

                    int diff_rem = crm[i] - crm[j];
                    int diff_sz = chc[i] - chc[j];
                    if (diff_sz == 0 || diff_rem < 0 || diff_rem > diff_sz) continue;

                    if (diff_rem == 0 || diff_rem == diff_sz) {
                        for (int a = 0; a < chc[i]; a++) {
                            int cell = ch[i * 8 + a];
                            int in_j = 0;
                            for (int b = 0; b < chc[j]; b++) {
                                if (cell == ch[j * 8 + b]) { in_j = 1; break; }
                            }
                            if (!in_j) {
                                int word = cell >> 5, bit = cell & 31;
                                if (!(seen[word] & (1u << bit))) {
                                    seen[word] |= (1u << bit);
                                    if (diff_rem == 0)
                                        out_safe[safe_count++] = cell;
                                    else
                                        out_mines[mine_count++] = cell;
                                    changed = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return (mine_count << 16) | (safe_count & 0xFFFF);
}

__device__ __forceinline__ int ai_choose_sh(int rows, int cols,
                                              int* revealed, int* flagged,
                                              const int* mine_cells,
                                              int* work_buf, int* mine_buf,
                                              int* out_cell,
                                              uint32_t* seen) {
    int result = infer_all_sh_iterative(rows, cols, revealed, flagged, mine_cells,
                                        work_buf, mine_buf, seen);
    int mine_count = result >> 16;
    int safe_count = result & 0xFFFF;

    for (int i = 0; i < mine_count; i++) {
        int mi = mine_buf[i];
        if (!flagged[mi]) { *out_cell = mi; return 0; }
    }

    for (int i = 0; i < safe_count; i++) {
        int si = work_buf[i];
        if (!revealed[si]) { *out_cell = si; return 1; }
    }

    int total = rows * cols;
    int cand_count = 0;
    int total_words = (total + 31) / 32;
    for (int w = 0; w < total_words; w++) seen[w] = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            if (!revealed[idx]) continue;
            for (int i = 0; i < 8; i++) {
                int nr = r + dr[i], nc = c + dc[i];
                if (!in_bounds(nr, nc, rows, cols)) continue;
                int ni = nr * cols + nc;
                if (revealed[ni] || flagged[ni]) continue;
                int word = ni >> 5, bit = ni & 31;
                if (!(seen[word] & (1u << bit))) {
                    seen[word] |= (1u << bit);
                    work_buf[cand_count++] = ni;
                }
            }
        }
    }

    int best = -1;
    float min_p = 1.0f;

    for (int ci = 0; ci < cand_count; ci++) {
        int i = work_buf[ci];
        int r = i / cols, c = i % cols;
        float sum = 0.0f, cnt = 0.0f;

        for (int k = 0; k < 8; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (!in_bounds(nr, nc, rows, cols)) continue;
            int ni = nr * cols + nc;
            if (!revealed[ni]) continue;

            int n = 0, fc = 0, hc = 0;
            for (int j = 0; j < 8; j++) {
                int nnr = nr + dr[j], nnc = nc + dc[j];
                if (!in_bounds(nnr, nnc, rows, cols)) continue;
                int nni = nnr * cols + nnc;
                n += mine_cells[nni];
                if (flagged[nni]) fc++;
                else if (!revealed[nni]) hc++;
            }
            if (hc > 0) {
                sum += max(0, n - fc) / (float)hc;
                cnt += 1.0f;
            }
        }

        float p = cnt > 0 ? max(0.0f, min(1.0f, sum / cnt)) : 0.5f;
        if (p < min_p) { min_p = p; best = i; }
    }
    *out_cell = best;
    return 1;
}

__device__ __forceinline__ int run_one_game_sh(int rows, int cols, int mines, int seed,
                                                 int* w_revealed, int* w_flagged, int* w_mines,
                                                 int* w_qr, int* w_qc, int* w_all, int* w_inf,
                                                 uint32_t* w_seen,
                                                 int* out_steps, int* out_flags) {
    int total = rows * cols;
    int steps = 0, flags_used = 0, done = 0, won = 0;
    int safe_remaining = total - mines;

    for (int i = 0; i < total; i++) { w_revealed[i] = 0; w_flagged[i] = 0; }

    int first_r = seed % rows;
    int first_c = (seed / rows) % cols;
    place_mines_sh(rows, cols, mines, first_r, first_c, w_mines, seed, w_all);
    do_reveal_sh(first_r, first_c, rows, cols, w_revealed, w_flagged, w_mines, w_qr, w_qc, &safe_remaining);
    steps++;

    if (safe_remaining == 0) { won = 1; done = 1; }

    int limit = total * 5;
    while (!done && steps < limit) {
        int target = -1;
        int action = ai_choose_sh(rows, cols, w_revealed, w_flagged, w_mines,
                                   w_inf, w_all, &target, w_seen);
        if (target < 0 || target >= total) break;

        if (action == 0) {
            w_flagged[target] = 1;
            flags_used++;
        } else {
            if (w_mines[target]) {
                w_revealed[target] = 1; done = 1; won = 0;
            } else {
                do_reveal_sh(target / cols, target % cols, rows, cols, w_revealed, w_flagged, w_mines,
                              w_qr, w_qc, &safe_remaining);
                if (safe_remaining == 0) { done = 1; won = 1; }
            }
        }
        if (done) break;
        steps++;
    }
    *out_steps = steps;
    *out_flags = flags_used;
    return won;
}

#endif // MINESWEEPER_GAME_H
