from unittest import TestCase

import numpy as np

from alab.hash import MinHash


class TestMinHash(TestCase):
    def test_create_signature(self):
        k = 256
        mh = MinHash(256)
        hashes = mh.create_signature([1, 2, 3])
        self.assertEqual(k, len(hashes))
        hashes = mh.create_signature(np.random.normal(0, 100000, [5000]))
        self.assertEqual(k, len(hashes))
        hashes = mh.create_signature([])
        self.assertEqual(k, len(hashes))
        self.assertTrue(all(hashes == np.zeros([k])))

    def test_calc_similarity(self):
        max_cardinality = 1000
        k = 256
        mh = MinHash(k)

        u = np.array(list(range(max_cardinality)))
        u_set = set(u)
        u_sign = mh.create_signature(u)
        for i in range(0, max_cardinality+1, 100):
            v = np.array(list(range(i)))
            v_sign = mh.create_signature(v)

            mh_sim = mh.calc_similarity(u_sign, v_sign)
            org_sim = len(u_set.intersection(v)) / len(u_set.union(v))
            self.assertTrue(abs(mh_sim-org_sim) < 0.1)
