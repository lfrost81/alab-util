from unittest import TestCase
from alab.share import singleton


class TestSingleton(TestCase):
    def test_singleton(self):
        self.assertTrue(TmpClass.value, 150)


@singleton()
class TmpClass():
    def __init__(self):
        self.value = 150


