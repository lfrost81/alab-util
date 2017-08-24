import numpy as np
import xxhash


class MinHash:
    def __init__(self, k, seed=1981):
        self.k = k
        self.seed = seed
        return

    def create_signature(self, iterable):
        if len(iterable) == 0:
            return [0 for _ in range(self.k)]
        return np.array([min(self.__permute_set(iterable, i)) for i in range(self.k)],
                        dtype=np.uint32)

    def calc_similarity(self, a, b):
        return len(list(filter(lambda x: x, [ae == be for ae, be in zip(a, b)]))) / self.k

    def __permute_set(self, iterable, nth):
        return [xxhash.xxh32(str(item), seed=self.seed+nth).intdigest() for item in iterable]


class FastMinHash:
    def __init__(self, k, seed=1981):
        self.k = k
        self.seed = seed
        return

    def create_signature(self, iterable):
        if len(iterable) == 0:
            return np.array([0 for _ in range(self.k)], dtype=np.uint32)

        hashes = [xxhash.xxh32(str(item), seed=self.seed).intdigest() for item in iterable]
        hashes_len = len(hashes)
        sorted_hashes = sorted(hashes)
        if hashes_len < self.k:
            for i in range(self.k - hashes_len):
                sorted_hashes.append(sorted_hashes[i % hashes_len])
            return sorted_hashes

        return sorted_hashes[0:self.k]

    def calc_similarity(self, a, b):
        if type(a) is set:
            return len(a.intersection(b)) / self.k
        elif type(b) is set:
            return len(b.intersection(a)) / self.k
        else:
            return len(set(a).intersection(b)) / self.k


