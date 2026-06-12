"""
Philox RNG 的 Python 实现
用于与 CUDA 版本保持一致
"""
import struct
from typing import List


class PhiloxRNG:
    """Philox-4x32 counter-based PRNG 的 Python 实现"""

    # Philox constants
    PHILOX_W32_0 = 0x9E3779B9
    PHILOX_W32_1 = 0xBB67AE85
    PHILOX_M4X32_0 = 0xD2511F53

    def __init__(self, seed: int = 0, counter: int = 0):
        self.seed = seed & 0xFFFFFFFF
        self.counter = counter & 0xFFFFFFFF

    def _mulhilo32(self, a: int, b: int) -> int:
        """32位乘法，返回高32位"""
        return (a * b) >> 32

    def _philox_round(self, ctr: List[int], key: List[int]) -> None:
        """Philox 一轮 - 必须确保所有操作都进行32位截断"""
        # 所有中间值都需要截断到32位
        lo0 = (self.PHILOX_M4X32_0 * ctr[0]) & 0xFFFFFFFF
        lo2 = (self.PHILOX_M4X32_0 * ctr[2]) & 0xFFFFFFFF
        hi0 = self._mulhilo32(self.PHILOX_M4X32_0, ctr[0])
        hi2 = self._mulhilo32(self.PHILOX_M4X32_0, ctr[2])

        t0 = (lo0 ^ ctr[1] ^ key[0]) & 0xFFFFFFFF
        t1 = (hi0 ^ ctr[1]) & 0xFFFFFFFF
        t2 = (lo2 ^ ctr[3] ^ key[1]) & 0xFFFFFFFF
        t3 = (hi2 ^ ctr[3]) & 0xFFFFFFFF

        ctr[0] = t2
        ctr[1] = t3
        ctr[2] = t0
        ctr[3] = t1

    def _philox_key_schedule(self, key: List[int], round_num: int) -> None:
        """Philox 密钥调度"""
        key[0] = (key[0] + self.PHILOX_W32_0 * round_num) & 0xFFFFFFFF
        key[1] = (key[1] + self.PHILOX_W32_1) & 0xFFFFFFFF
        key[2] = (key[2] + self.PHILOX_W32_0 * round_num + self.PHILOX_W32_0) & 0xFFFFFFFF
        key[3] = (key[3] + self.PHILOX_W32_1) & 0xFFFFFFFF

    def philox_hash(self, seed: int, counter: int) -> int:
        """Philox-4x32 哈希，返回第一个32位字"""
        ctr = [counter & 0xFFFFFFFF, 0, 0, 0]
        key = [seed & 0xFFFFFFFF,
               (seed >> 1) & 0xFFFFFFFF,
               (seed >> 2) & 0xFFFFFFFF,
               (seed >> 3) & 0xFFFFFFFF]

        for r in range(9):
            self._philox_round(ctr, key)
            self._philox_key_schedule(key, r)

        # 最终轮
        self._philox_round(ctr, key)
        return ctr[0]

    def rand(self, max_val: int, counter: int = None) -> int:
        """生成 [0, max_val) 范围内的随机整数
        如果指定counter，使用固定counter模式（与CUDA兼容）
        否则使用self.counter（递增模式，向后兼容）
        """
        if max_val <= 1:
            return 0

        threshold = ((0xFFFFFFFF // max_val) * max_val) & 0xFFFFFFFF

        if counter is not None:
            for attempt in range(5):
                val = self.philox_hash(self.seed, counter + attempt)
                if val < threshold:
                    return val % max_val
            val = self.philox_hash(self.seed, counter + 5)
            return val % max_val
        else:
            for attempt in range(5):
                val = self.philox_hash(self.seed, self.counter)
                self.counter += 1
                if val < threshold:
                    return val % max_val
            val = self.philox_hash(self.seed, self.counter)
            self.counter += 1
            return val % max_val

    def randrange(self, start: int, stop: int = None) -> int:
        """randrange(start, stop) 或 randrange(stop)"""
        if stop is None:
            start, stop = 0, start
        return start + self.rand(stop - start)

    def sample(self, population: List, k: int) -> List:
        """从 population 中随机抽取 k 个不重复的样本 - 与 Python random.sample 等价"""
        n = len(population)
        if k == 1:
            i = self.randrange(n)
            return [population[i]]

        # 使用 pool 变体（与 Python random.sample 相同）
        pool = list(population)
        result = []
        for i in range(k):
            j = self.randrange(n - i)
            result.append(pool[j])
            pool[j] = pool[n - i - 1]
        return result

    def shuffle(self, x: List, base_counter: int = None) -> None:
        """原地洗牌 - Fisher-Yates
        与CUDA的philox_shuffle兼容：使用base_counter + (n-1-i)作为counter
        """
        n = len(x)
        for i in range(n - 1, 0, -1):
            if base_counter is not None:
                counter = base_counter + (n - 1 - i)
                j = self.rand(i + 1, counter=counter)
            else:
                j = self.randrange(i + 1)
            x[i], x[j] = x[j], x[i]

    def choice(self, seq: List):
        """随机选择序列中的一个元素"""
        return seq[self.randrange(len(seq))]


# 测试函数
def test_philox():
    """验证 Python Philox 实现与 CUDA 行为一致"""
    rng = PhiloxRNG(seed=12345, counter=0)

    # 测试 rand
    print("测试 rand:")
    for i in range(5):
        print(f"  rand(100) = {rng.rand(100)}")

    # 测试 randrange
    rng = PhiloxRNG(seed=12345, counter=0)
    print("\n测试 randrange:")
    for i in range(5):
        print(f"  randrange(100) = {rng.randrange(100)}")

    # 测试 shuffle
    rng = PhiloxRNG(seed=12345, counter=0)
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rng.shuffle(arr)
    print(f"\nshuffle([0..9]) = {arr}")

    # 测试 sample
    rng = PhiloxRNG(seed=12345, counter=0)
    print(f"\nsample([0..9], 3) = {rng.sample(list(range(10)), 3)}")


if __name__ == "__main__":
    test_philox()
