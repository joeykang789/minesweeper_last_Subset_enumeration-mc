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
            // For large boards, skip expensive subset enumeration (O(N²))
            // MC handles probability estimation much better for large boards
            int total_cells = rows * cols;
            if (total_cells > 400) continue;

            int num_c = 0;
            int* ch = out_safe + total_cells;      // reuse out_safe tail for temp arrays
            int* chc = ch + 512 * 8;
            int* crm = chc + 512;

            for (int rr = 0; rr < rows && num_c < 512; rr++) {
                for (int cc = 0; cc < cols && num_c < 512; cc++) {
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

__device__ __forceinline__ int monte_carlo_infer(
    int rows, int cols,
    const int* revealed, const int* flagged, const int* mine_cells,
    int total_mines,
    int* work_buf, int* mine_buf, uint32_t* seen,
    uint32_t seed, int step
) {
    int total = rows * cols;
    int total_words = (total + 31) / 32;

    // Phase 1: collect all unknown cells into work_buf[0..unk_count)
    int unk_count = 0;
    int flagged_count = 0;
    for (int i = 0; i < total; i++) {
        if (flagged[i]) flagged_count++;
        else if (!revealed[i]) work_buf[unk_count++] = i;
    }
    int remaining_mines = total_mines - flagged_count;
    if (unk_count == 0 || remaining_mines <= 0) return -1;

    // Phase 2: build border constraints (use work_buf + 2*total onward)
    int* perm = work_buf + total;          // temp shuffle buffer
    int* cons_cells = work_buf + 2 * total;
    int max_cons = (total * 2) / 10;
    int* cons_hc = cons_cells + max_cons * 8;
    int* cons_rem = cons_hc + max_cons;

    int num_cons = 0;
    for (int r = 0; r < rows && num_cons < max_cons; r++) {
        for (int c = 0; c < cols && num_cons < max_cons; c++) {
            int idx = r * cols + c;
            if (!revealed[idx]) continue;
            int n = 0, fc = 0, hc = 0;
            int hlist[8];
            for (int k = 0; k < 8; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (!in_bounds(nr, nc, rows, cols)) continue;
                int ni = nr * cols + nc;
                n += mine_cells[ni];
                if (flagged[ni]) fc++;
                else if (!revealed[ni]) { hlist[hc++] = ni; }
            }
            if (hc == 0) continue;
            int rem = n - fc;
            if (rem < 0 || rem > hc) continue;
            cons_hc[num_cons] = hc;
            cons_rem[num_cons] = rem;
            for (int k = 0; k < hc; k++)
                cons_cells[num_cons * 8 + k] = hlist[k];
            num_cons++;
        }
    }

    // Phase 3: separate frontier vs interior cells
    // First pass: count frontier cells (does NOT modify work_buf yet)
    int frontier_count = 0;
    int first_interior_cell = -1;
    for (int i = 0; i < unk_count; i++) {
        int cell = work_buf[i];
        int r = cell / cols, c = cell % cols;
        int adj_revealed = 0;
        for (int k = 0; k < 8 && !adj_revealed; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (in_bounds(nr, nc, rows, cols) && revealed[nr * cols + nc])
                adj_revealed = 1;
        }
        if (adj_revealed) {
            frontier_count++;
        } else if (first_interior_cell < 0) {
            first_interior_cell = cell;
        }
    }
    int interior_count = unk_count - frontier_count;

    // Adaptive sample count
    int num_samples = (total > 1000) ? 50 : (total > 400) ? 100 : 200;

    int valid_count = 0;
    for (int i = 0; i < total; i++) mine_buf[i] = 0;

    uint32_t mc_ctr = 1000000 + (uint32_t)step * 50000;
    int to_select = remaining_mines < unk_count ? remaining_mines : unk_count;

    // Decide which MC path to take
    int use_frontier_only = (frontier_count >= 50 && frontier_count < unk_count);

    if (frontier_count == 0) {
        // all cells are interior — uniform probability, pick any
        return (interior_count > 0) ? first_interior_cell : -1;
    }

    if (!use_frontier_only) {
        // Full MC path: work_buf[0..unk_count) still has all unknown cells (not compacted)
        for (int s = 0; s < num_samples; s++) {
            for (int i = 0; i < unk_count; i++) perm[i] = work_buf[i];
            for (int i = 0; i < to_select; i++) {
                int range = unk_count - i;
                uint32_t j = philox_rand(seed, mc_ctr, range);
                mc_ctr += 8;
                int tmp = perm[i]; perm[i] = perm[i + j]; perm[i + j] = tmp;
            }

            for (int w = 0; w < total_words; w++) seen[w] = 0;
            for (int i = 0; i < to_select; i++) {
                int cell = perm[i];
                seen[cell >> 5] |= (1u << (cell & 31));
            }

            int valid = 1;
            for (int ci = 0; ci < num_cons && valid; ci++) {
                int cnt = 0;
                for (int k = 0; k < cons_hc[ci]; k++) {
                    int cell = cons_cells[ci * 8 + k];
                    cnt += (seen[cell >> 5] >> (cell & 31)) & 1;
                }
                if (cnt != cons_rem[ci]) valid = 0;
            }

            if (valid) {
                valid_count++;
                for (int i = 0; i < to_select; i++)
                    mine_buf[perm[i]]++;
            }
        }
    } else {
        // Frontier-only path: compact frontier cells to work_buf[0..frontier_count)
        int write_idx = 0;
        for (int i = 0; i < unk_count; i++) {
            int cell = work_buf[i];
            int r = cell / cols, c = cell % cols;
            int adj_revealed = 0;
            for (int k = 0; k < 8 && !adj_revealed; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (in_bounds(nr, nc, rows, cols) && revealed[nr * cols + nc])
                    adj_revealed = 1;
            }
            if (adj_revealed) work_buf[write_idx++] = cell;
        }

        int expected_k = (frontier_count * remaining_mines + unk_count / 2) / unk_count;
        int min_k = max(0, remaining_mines - interior_count);
        int max_k = min(remaining_mines, frontier_count);
        if (expected_k < min_k) expected_k = min_k;
        if (expected_k > max_k) expected_k = max_k;
        if (expected_k > frontier_count) expected_k = frontier_count;

        for (int s = 0; s < num_samples; s++) {
            // copy frontier list to perm, shuffle in-place
            for (int i = 0; i < frontier_count; i++) perm[i] = work_buf[i];
            for (int i = 0; i < expected_k; i++) {
                int range = frontier_count - i;
                uint32_t j = philox_rand(seed, mc_ctr, range);
                mc_ctr += 8;
                int tmp = perm[i]; perm[i] = perm[i + j]; perm[i + j] = tmp;
            }

            for (int w = 0; w < total_words; w++) seen[w] = 0;
            for (int i = 0; i < expected_k; i++) {
                int cell = perm[i];
                seen[cell >> 5] |= (1u << (cell & 31));
            }

            int valid = 1;
            for (int ci = 0; ci < num_cons && valid; ci++) {
                int cnt = 0;
                for (int k = 0; k < cons_hc[ci]; k++) {
                    int cell = cons_cells[ci * 8 + k];
                    cnt += (seen[cell >> 5] >> (cell & 31)) & 1;
                }
                if (cnt != cons_rem[ci]) valid = 0;
            }

            if (valid) {
                valid_count++;
                for (int i = 0; i < expected_k; i++)
                    mine_buf[perm[i]]++;
            }
        }
    }

    // Phase 4: find best cell (lowest mine probability)
    int best = -1;
    float min_p = 1.0f;
    if (valid_count > 0) {
        for (int i = 0; i < frontier_count; i++) {
            int cell = work_buf[i];
            float p = (float)mine_buf[cell] / (float)valid_count;
            if (p < min_p) { min_p = p; best = cell; }
        }

        // interior cells: uniform background probability
        float bg_p = (float)remaining_mines / (float)unk_count;
        if (bg_p < min_p && first_interior_cell >= 0) {
            best = first_interior_cell;
        }
    }

    return best;
}

__device__ __forceinline__ int mcmc_infer(
    int rows, int cols,
    const int* revealed, const int* flagged, const int* mine_cells,
    int total_mines,
    int* work_buf, int* mine_buf, uint32_t* seen,
    uint32_t seed, int step
) {
    int total = rows * cols;

    // Step 1: collect unknown cells and count flagged
    int unk_count = 0;
    int flagged_count = 0;
    for (int i = 0; i < total; i++) {
        if (flagged[i]) flagged_count++;
        else if (!revealed[i]) work_buf[unk_count++] = i;
    }
    int remaining_mines = total_mines - flagged_count;
    if (unk_count == 0 || remaining_mines <= 0) return -1;
    int to_select = remaining_mines < unk_count ? remaining_mines : unk_count;

    // Step 2: build border constraints (same as MC)
    int num_cons = 0;
    int cons_cells[512 * 8];
    int cons_hc[512];
    int cons_rem[512];

    for (int r = 0; r < rows && num_cons < 512; r++) {
        for (int c = 0; c < cols && num_cons < 512; c++) {
            int idx = r * cols + c;
            if (!revealed[idx]) continue;
            int n = 0, fc = 0, hc = 0;
            int hlist[8];
            for (int k = 0; k < 8; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (!in_bounds(nr, nc, rows, cols)) continue;
                int ni = nr * cols + nc;
                n += mine_cells[ni];
                if (flagged[ni]) fc++;
                else if (!revealed[ni]) hlist[hc++] = ni;
            }
            if (hc == 0) continue;
            int rem = n - fc;
            if (rem < 0 || rem > hc) continue;
            cons_hc[num_cons] = hc;
            cons_rem[num_cons] = rem;
            for (int k = 0; k < hc; k++)
                cons_cells[num_cons * 8 + k] = hlist[k];
            num_cons++;
        }
    }

    // Step 3: build cell-to-constraint mapping
    int cell_cons_cnt[MAX_BOARD_CELLS];
    int cell_cons[MAX_BOARD_CELLS * 8];
    for (int i = 0; i < total; i++) cell_cons_cnt[i] = 0;

    for (int ci = 0; ci < num_cons; ci++) {
        for (int k = 0; k < cons_hc[ci]; k++) {
            int cell = cons_cells[ci * 8 + k];
            cell_cons[cell * 8 + cell_cons_cnt[cell]] = ci;
            cell_cons_cnt[cell]++;
        }
    }

    // Step 4: find initial valid config via rejection sampling (up to 200 attempts)
    int perm[MAX_BOARD_CELLS];
    uint32_t mc_ctr = 1000000 + (uint32_t)step * 50000;

    int is_mine[MAX_BOARD_CELLS];
    int mine_list[MAX_BOARD_CELLS];
    int free_list[MAX_BOARD_CELLS];
    int num_mines = 0, num_free = 0;

    int found = 0;
    for (int attempt = 0; attempt < 200; attempt++) {
        for (int i = 0; i < unk_count; i++) perm[i] = work_buf[i];
        for (int i = 0; i < to_select; i++) {
            int range = unk_count - i;
            uint32_t j = philox_rand(seed, mc_ctr, range);
            mc_ctr += 8;
            int tmp = perm[i]; perm[i] = perm[i + j]; perm[i + j] = tmp;
        }

        for (int i = 0; i < total; i++) is_mine[i] = 0;
        for (int i = 0; i < to_select; i++) is_mine[perm[i]] = 1;

        int valid = 1;
        for (int ci = 0; ci < num_cons && valid; ci++) {
            int cnt = 0;
            for (int k = 0; k < cons_hc[ci]; k++) {
                int cell = cons_cells[ci * 8 + k];
                cnt += is_mine[cell];
            }
            if (cnt != cons_rem[ci]) valid = 0;
        }

        if (valid) {
            found = 1;
            break;
        }
    }
    if (!found) return -1;

    num_mines = 0; num_free = 0;
    for (int i = 0; i < unk_count; i++) {
        int cell = work_buf[i];
        if (is_mine[cell]) mine_list[num_mines++] = cell;
        else free_list[num_free++] = cell;
    }

    uint32_t chain_ctr = mc_ctr + 100000;

    // Step 5: burn-in — 100 swap perturbations (no recording)
    for (int s = 0; s < 100; s++) {
        uint32_t mi = philox_rand(seed, chain_ctr, num_mines);
        chain_ctr += 8;
        int mc = mine_list[mi];

        uint32_t fi = philox_rand(seed, chain_ctr, num_free);
        chain_ctr += 8;
        int fc_cell = free_list[fi];

        // propose swap
        is_mine[mc] = 0;
        is_mine[fc_cell] = 1;

        // collect affected constraints (union of both cells' constraints)
        int affected[16], aff_cnt = 0;
        for (int k = 0; k < cell_cons_cnt[mc]; k++) {
            int ci = cell_cons[mc * 8 + k];
            int dup = 0;
            for (int a = 0; a < aff_cnt; a++) if (affected[a] == ci) { dup = 1; break; }
            if (!dup) affected[aff_cnt++] = ci;
        }
        for (int k = 0; k < cell_cons_cnt[fc_cell]; k++) {
            int ci = cell_cons[fc_cell * 8 + k];
            int dup = 0;
            for (int a = 0; a < aff_cnt; a++) if (affected[a] == ci) { dup = 1; break; }
            if (!dup) affected[aff_cnt++] = ci;
        }

        int valid = 1;
        for (int a = 0; a < aff_cnt && valid; a++) {
            int ci = affected[a];
            int cnt = 0;
            for (int k = 0; k < cons_hc[ci]; k++) {
                int cell = cons_cells[ci * 8 + k];
                cnt += is_mine[cell];
            }
            if (cnt != cons_rem[ci]) valid = 0;
        }

        if (valid) {
            mine_list[mi] = fc_cell;
            free_list[fi] = mc;
        } else {
            is_mine[mc] = 1;
            is_mine[fc_cell] = 0;
        }
    }

    // Step 6: sampling — 200 swap perturbations (record mine counts after each)
    int hit_count[MAX_BOARD_CELLS];
    for (int i = 0; i < total; i++) hit_count[i] = 0;

    for (int s = 0; s < 200; s++) {
        uint32_t mi = philox_rand(seed, chain_ctr, num_mines);
        chain_ctr += 8;
        int mc = mine_list[mi];

        uint32_t fi = philox_rand(seed, chain_ctr, num_free);
        chain_ctr += 8;
        int fc_cell = free_list[fi];

        is_mine[mc] = 0;
        is_mine[fc_cell] = 1;

        int affected[16], aff_cnt = 0;
        for (int k = 0; k < cell_cons_cnt[mc]; k++) {
            int ci = cell_cons[mc * 8 + k];
            int dup = 0;
            for (int a = 0; a < aff_cnt; a++) if (affected[a] == ci) { dup = 1; break; }
            if (!dup) affected[aff_cnt++] = ci;
        }
        for (int k = 0; k < cell_cons_cnt[fc_cell]; k++) {
            int ci = cell_cons[fc_cell * 8 + k];
            int dup = 0;
            for (int a = 0; a < aff_cnt; a++) if (affected[a] == ci) { dup = 1; break; }
            if (!dup) affected[aff_cnt++] = ci;
        }

        int valid = 1;
        for (int a = 0; a < aff_cnt && valid; a++) {
            int ci = affected[a];
            int cnt = 0;
            for (int k = 0; k < cons_hc[ci]; k++) {
                int cell = cons_cells[ci * 8 + k];
                cnt += is_mine[cell];
            }
            if (cnt != cons_rem[ci]) valid = 0;
        }

        if (valid) {
            mine_list[mi] = fc_cell;
            free_list[fi] = mc;
        } else {
            is_mine[mc] = 1;
            is_mine[fc_cell] = 0;
        }

        for (int i = 0; i < num_mines; i++)
            hit_count[mine_list[i]]++;
    }

    // Step 7: pick cell with lowest mine probability
    int best = -1;
    float min_p = 1.0f;
    for (int i = 0; i < unk_count; i++) {
        int cell = work_buf[i];
        float p = (float)hit_count[cell] / 200.0f;
        if (p < min_p) { min_p = p; best = cell; }
    }
    return best;
}

__device__ __forceinline__ int ai_choose_sh(int rows, int cols,
                                              int* revealed, int* flagged,
                                              const int* mine_cells,
                                              int total_mines,
                                              int* work_buf, int* mine_buf,
                                              int* out_cell,
                                              uint32_t* seen,
                                              uint32_t seed, int step) {
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

    int mc_best = monte_carlo_infer(rows, cols, revealed, flagged, mine_cells,
                                     total_mines, work_buf, mine_buf, seen, seed, step);
    if (mc_best >= 0) {
        *out_cell = mc_best;
        return 1;
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
                                   mines, w_inf, w_all, &target, w_seen,
                                   seed, steps);
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
