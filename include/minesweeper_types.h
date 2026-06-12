#ifndef MINESWEEPER_TYPES_H
#define MINESWEEPER_TYPES_H

// Maximum board dimensions
// Reduced to 256 to minimize register usage and prevent spilling
// Boards > 256 cells are not supported in the small kernel path
#define MAX_BOARD_CELLS 256
#define MAX_BOARD_DIM   64
#define MAX_NEIGHBORS   8

// Threshold for small board vs large board kernel selection
#define SMALL_BOARD_THRESHOLD 256

// Maximum number of density configs merged in one kernel launch
#define MAX_MERGED_DENSITIES 32

// Game state flags
#define CELL_UNREVEALED  0
#define CELL_REVEALED    1
#define CELL_FLAGGED     2

// Result codes
#define RESULT_LOSE 0
#define RESULT_WIN  1

struct GameResult {
    int won;
    int steps;
    int flags_used;
};

#endif // MINESWEEPER_TYPES_H
