from alab.train.nn import DNNEstimator
from alab.data import NumpyDataFeeder
from unittest import TestCase

import tensorflow as tf
import os
import json


class TestDNNEstimator(TestCase):
    conf = {
        'epoch': 500,
        'learning_rate': 0.01,
        'cost_func': 'cross_entropy',
        'softmax': True,
        'optimizer': 'adam',
        'l2reg_lambda': 0,
        'initializer': 'xavier',
        'hidden_layers': [
            {
                'dim': 100,
                'activate_func': 'relu',
                'dropout': 0.9
            }, {
                'dim': 50,
                'activate_func': 'relu',
                'dropout': 0.9
            }
        ]
    }

    def test_fit(self):
        x = [[0, 0], [0, 1], [1, 0], [1,1]]
        y = [[0], [1], [1], [0]]

        train = DNNEstimator(self.conf, verbosity=0)
        train.fit(x, y)
        accuracy = train.evaluate(x, y)
        self.assertGreater(accuracy, 0.5)

    def test_fit_batch(self):
        x = [[0, 0], [0, 1], [1, 0], [1,1]]
        y = [[0], [1], [1], [0]]

        feeder = NumpyDataFeeder(fetch_size=4)
        feeder.add_data('x', x)
        feeder.add_data('y', y)

        train = DNNEstimator(self.conf, verbosity=0)

        train.fit_batch(feeder)
        accuracy = train.evaluate(x, y)
        self.assertGreater(accuracy, 0.5)


