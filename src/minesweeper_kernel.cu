#include "minesweeper_types.h"
#include "minesweeper_game.h"

__global__ void minesweeper_kernel_small(
    int rows, int cols, int mines,
    int base_seed, int games_per_thread, int max_active_threads,
    int* results_win, int* results_steps, int* results_flags
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max_active_threads) return;

    int seed_base = base_seed + tid * 100000;
    int total = rows * cols;

    int revealed[MAX_BOARD_CELLS];
    int flagged[MAX_BOARD_CELLS];
    int mine_cells[MAX_BOARD_CELLS];
    int q_r[MAX_BOARD_CELLS];
    int q_c[MAX_BOARD_CELLS];
    int all_cells[MAX_BOARD_CELLS];
    int inf_buf[MAX_BOARD_CELLS * 2];
    uint32_t seen[(MAX_BOARD_CELLS + 31) / 32];

    int wins = 0, steps = 0, flags = 0;
    for (int g = 0; g < games_per_thread; g++) {
        int s, f;
        if (run_one_game_sh(rows, cols, mines, seed_base + g,
                              revealed, flagged, mine_cells,
                              q_r, q_c, all_cells, inf_buf,
                              seen, &s, &f)) {
            wins++;
        }
        steps += s;
        flags += f;
    }

    results_win[tid] = wins;
    results_steps[tid] = steps;
    results_flags[tid] = flags;
}

__global__ void minesweeper_kernel_large(
    int rows, int cols, int mines,
    int base_seed, int games_per_thread, int max_active_threads,
    int* w_revealed, int* w_flagged, int* w_mines,
    int* w_qr, int* w_qc, int* w_all, int* w_inf, uint32_t* w_seen,
    int* results_win, int* results_steps, int* results_flags
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max_active_threads) return;

    int total = rows * cols;
    int seed_base = base_seed + tid * 100000;
    int off = tid * total;

    int wins = 0, steps = 0, flags = 0;
    for (int g = 0; g < games_per_thread; g++) {
        int s, f;
        if (run_one_game_sh(rows, cols, mines, seed_base + g,
                              w_revealed + off, w_flagged + off, w_mines + off,
                              w_qr + off, w_qc + off, w_all + off, w_inf + off,
                              w_seen + (off >> 5), &s, &f)) {
            wins++;
        }
        steps += s;
        flags += f;
    }

    results_win[tid] = wins;
    results_steps[tid] = steps;
    results_flags[tid] = flags;
}

__global__ void minesweuper_super_merged_kernel(
    int num_configs,
    const int* config_rows,
    const int* config_cols,
    const int* config_mines,
    const int* config_seeds,
    const int* config_prefix_games,
    const int* config_trials,
    const int* config_cell_offsets,
    int total_games,
    int* w_revealed, int* w_flagged, int* w_mines,
    int* w_qr, int* w_qc, int* w_all, int* w_inf, uint32_t* w_seen,
    int* out_win, int* out_steps, int* out_flags
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_games) return;

    int cfg_idx = 0;
    {
        int lo = 0, hi = num_configs - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (gid < config_prefix_games[mid]) hi = mid;
            else lo = mid + 1;
        }
        cfg_idx = lo;
    }

    int games_before_cfg = (cfg_idx > 0) ? config_prefix_games[cfg_idx - 1] : 0;
    int game_in_cfg = gid - games_before_cfg;

    // 先检查 game_in_cfg 是否有效，避免无效的 cell_offset 计算
    if (game_in_cfg < 0 || game_in_cfg >= config_trials[cfg_idx]) return;

    int rows = config_rows[cfg_idx];
    int cols = config_cols[cfg_idx];
    int mines = config_mines[cfg_idx];
    int seed = config_seeds[cfg_idx] + game_in_cfg;
    int total = rows * cols;

    int cell_offset = config_cell_offsets[cfg_idx] + total * game_in_cfg;

    int inf_offset = cell_offset * 4;

    int s, f;
    int won = run_one_game_sh(rows, cols, mines, seed,
                               w_revealed + cell_offset, w_flagged + cell_offset, w_mines + cell_offset,
                               w_qr + cell_offset, w_qc + cell_offset, w_all + cell_offset, w_inf + inf_offset,
                               w_seen + (cell_offset >> 5),
                               &s, &f);

    out_win[gid] = won;
    out_steps[gid] = s;
    out_flags[gid] = f;
}

extern "C" __declspec(dllexport) void launch_minesweeper_kernel(
    int num_blocks, int threads_per_block,
    int rows, int cols, int mines,
    int base_seed, int games_per_thread,
    int max_active_threads,
    int use_small_board,
    int* d_work_revealed,
    int* d_work_flagged,
    int* d_work_mine_cells,
    int* d_work_qr,
    int* d_work_qc,
    int* d_work_all,
    int* d_work_inf,
    uint32_t* d_work_seen,
    int* d_results_win,
    int* d_results_steps,
    int* d_results_flags
) {
    if (use_small_board) {
        minesweeper_kernel_small<<<num_blocks, threads_per_block>>>(
            rows, cols, mines, base_seed, games_per_thread, max_active_threads,
            d_results_win, d_results_steps, d_results_flags
        );
    } else {
        minesweeper_kernel_large<<<num_blocks, threads_per_block>>>(
            rows, cols, mines, base_seed, games_per_thread, max_active_threads,
            d_work_revealed, d_work_flagged, d_work_mine_cells,
            d_work_qr, d_work_qc, d_work_all, d_work_inf, d_work_seen,
            d_results_win, d_results_steps, d_results_flags
        );
    }
}

extern "C" __declspec(dllexport) void launch_merged_minesweeper_kernel(
    int num_blocks, int threads_per_block,
    int rows, int cols,
    int num_configs,
    const int* d_config_mines,
    const int* d_config_seeds,
    const int* d_config_games_per_cfg,
    int total_games,
    int use_small_board,
    int* d_work_revealed,
    int* d_work_flagged,
    int* d_work_mine_cells,
    int* d_work_qr,
    int* d_work_qc,
    int* d_work_all,
    int* d_work_inf,
    uint32_t* d_work_seen,
    int* d_out_win,
    int* d_out_steps,
    int* d_out_flags
) {
    // Stub - not used
}

extern "C" __declspec(dllexport) void launch_super_merged_minesweeper_kernel(
    int num_blocks, int threads_per_block,
    int num_configs,
    const int* d_config_rows,
    const int* d_config_cols,
    const int* d_config_mines,
    const int* d_config_seeds,
    const int* d_config_prefix_games,
    const int* d_config_trials,
    const int* d_config_cell_offsets,
    int total_games,
    int* d_work_revealed,
    int* d_work_flagged,
    int* d_work_mine_cells,
    int* d_work_qr,
    int* d_work_qc,
    int* d_work_all,
    int* d_work_inf,
    uint32_t* d_work_seen,
    int* d_out_win,
    int* d_out_steps,
    int* d_out_flags
) {
    minesweuper_super_merged_kernel<<<num_blocks, threads_per_block>>>(
        num_configs,
        d_config_rows, d_config_cols, d_config_mines, d_config_seeds, d_config_prefix_games, d_config_trials, d_config_cell_offsets,
        total_games,
        d_work_revealed, d_work_flagged, d_work_mine_cells,
        d_work_qr, d_work_qc, d_work_all, d_work_inf, d_work_seen,
        d_out_win, d_out_steps, d_out_flags
    );
}
