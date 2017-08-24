from unittest import TestCase
from . import NumpyFeedingElement
import numpy as np


class TestNumpyFeedingElement(TestCase):
    def test_feed_data(self):
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

        nfe = NumpyFeedingElement(x)
        result = nfe.feed_data(2)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[0, 1], 2)

        nfe.rewind()
        result = nfe.feed_data(2)
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 2], 6)
        self.assertEqual(len(nfe.feed_data(2)), 1)

    def test_row_len(self):
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])

        nfe = NumpyFeedingElement(x)
        self.assertEqual(nfe.row_len(), 2)

    def test_col_len(self):
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])

        nfe = NumpyFeedingElement(x)
        self.assertEqual(nfe.col_len(), 3)
