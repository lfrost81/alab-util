from unittest import TestCase
from . import NumpyDataFeeder
import numpy as np


class TestNumpyDataFeeder(TestCase):
    def test_add_data(self):
        ndf = NumpyDataFeeder(2)
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])

        ndf.add_data('x', x)

        self.assertNotEquals(None, ndf.elements['x'])

    def test_feed(self):
        ndf = NumpyDataFeeder(2)
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])

        ndf.add_data('x', x)
        result = ndf.feed('x')
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 2], 6)

        ndf.rewind()
        result = ndf.feed('x')
        self.assertEqual(result[0, 1], 2)
        self.assertEqual(result[1, 2], 6)

        ndf = NumpyDataFeeder(3)
        ndf.add_data('x', x)
        result = ndf.feed('x')
        self.assertEqual(len(result), 2)
