#ifndef PHILOX_RNG_H
#define PHILOX_RNG_H

#include <cuda_runtime.h>

// Philox-4x32 counter-based PRNG (10 rounds)
// Reference: "Parallel Random Numbers: As Easy as 1, 2, 3"
// No state needed, just seed + counter. Perfect for saving registers on GPU.

// Philox constants
#define PHILOX_W32_0 0x9E3779B9U
#define PHILOX_W32_1 0xBB67AE85U
#define PHILOX_M4X32_0 0xD2511F53U

// 32-bit multiply, return high 32 bits of 64-bit product
__device__ __forceinline__ uint32_t _mulhilo32(uint32_t a, uint32_t b) {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

// Single Philox-4x32 round
__device__ __forceinline__ void _philox_round(uint32_t ctr[4], const uint32_t key[4]) {
    uint32_t lo0 = PHILOX_M4X32_0 * ctr[0];
    uint32_t lo2 = PHILOX_M4X32_0 * ctr[2];
    uint32_t hi0 = _mulhilo32(PHILOX_M4X32_0, ctr[0]);
    uint32_t hi2 = _mulhilo32(PHILOX_M4X32_0, ctr[2]);

    uint32_t t0 = lo0 ^ ctr[1] ^ key[0];
    uint32_t t1 = hi0 ^ ctr[1];
    uint32_t t2 = lo2 ^ ctr[3] ^ key[1];
    uint32_t t3 = hi2 ^ ctr[3];

    ctr[0] = t2;
    ctr[1] = t3;
    ctr[2] = t0;
    ctr[3] = t1;
}

// Key schedule: add round-dependent constants
__device__ __forceinline__ void _philox_key_schedule(uint32_t key[4], uint32_t round) {
    key[0] += PHILOX_W32_0 * round;
    key[1] += PHILOX_W32_1;
    key[2] += PHILOX_W32_0 * round + PHILOX_W32_0;
    key[3] += PHILOX_W32_1;
}

// Philox-4x32 hash: 10 rounds, returns first 32-bit word
__device__ __forceinline__ uint32_t philox_hash(uint32_t seed, uint32_t counter) {
    uint32_t ctr[4] = {counter, 0, 0, 0};
    uint32_t key[4] = {seed, seed >> 1, seed >> 2, seed >> 3};

    for (int r = 0; r < 9; r++) {
        _philox_round(ctr, key);
        _philox_key_schedule(key, r);
    }
    // Final round without key schedule
    _philox_round(ctr, key);

    return ctr[0];
}

// Generate random uint32 in [0, max_value) using rejection sampling
__device__ __forceinline__ uint32_t philox_rand(uint32_t seed, uint32_t counter, uint32_t max_val) {
    if (max_val <= 1) return 0;

    uint32_t threshold = ((~(uint32_t)0) / max_val) * max_val;

    // Try up to 5 attempts (extremely unlikely to fail)
    for (int attempt = 0; attempt < 5; attempt++) {
        uint32_t val = philox_hash(seed, counter + attempt);
        if (val < threshold) {
            return val % max_val;
        }
    }
    // Fallback (should almost never happen)
    return philox_hash(seed, counter + 5) % max_val;
}

// Fisher-Yates shuffle using Philox RNG
__device__ __forceinline__ void philox_shuffle(int* arr, int n, uint32_t seed, uint32_t base_counter) {
    for (int i = n - 1; i > 0; i--) {
        uint32_t j = philox_rand(seed, base_counter + (n - 1 - i), i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

#endif // PHILOX_RNG_H
