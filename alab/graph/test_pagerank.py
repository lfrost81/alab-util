from alab.graph.pagerank import rank_for_bipartite
from alab.graph.pagerank import rank
from unittest import TestCase

import numpy as np


class TestPageRank(TestCase):
    adj_mat = np.array([[0, 0, 0, 2, 12],
                        [0, 0, 0, 5, 12],
                        [0, 0, 0, 5, 12],
                        [2, 12, 12, 0, 0],
                        [15, 10, 10, 0, 0]])

    def test_rank(self):
        self.assertAlmostEqual(np.sum(rank(self.adj_mat)), 1, delta=.1)

    def test_rank_for_bipartite(self):
        r1, r2 = rank_for_bipartite(self.adj_mat, border=3)
        self.assertEqual(np.sum(r1), 1)
        self.assertEqual(np.sum(r2), 1)

