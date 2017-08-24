from alab.train.nn import DNNEstimator
from alab.data import NumpyDataFeeder
from unittest import TestCase


class TestDNNEstimator(TestCase):
    def test_fit(self):
        x = [[0, 0], [0, 1], [1, 0], [1,1]]
        y = [[0], [1], [1], [0]]

        train = DNNEstimator(hidden_layers=[200, 50], dropouts=[0.9, 0.9],
                             l2reg_lambda=0.001, epoch=10, learning_rate=0.01, initializer='xavier',
                             activate_func='relu', cost_func='cross_entropy', softmax=True,
                             optimizer='adam', verbosity=0)

        train.fit(x, y)
        accuracy = train.evaluate(x, y)
        self.assertGreater(accuracy, 0.5)

    def test_fit_batch(self):
        x = [[0, 0], [0, 1], [1, 0], [1,1]]
        y = [[0], [1], [1], [0]]

        feeder = NumpyDataFeeder(fetch_size=4)
        feeder.add_data('x', x)
        feeder.add_data('y', y)

        train = DNNEstimator(hidden_layers=[200, 50], dropouts=[0.9, 0.9],
                             l2reg_lambda=0.001, epoch=10, learning_rate=0.01, initializer='xavier',
                             activate_func='relu', cost_func='cross_entropy', softmax=True,
                             optimizer='adam', verbosity=0)

        train.fit_batch(feeder)
        accuracy = train.evaluate(x, y)
        self.assertGreater(accuracy, 0.5)
