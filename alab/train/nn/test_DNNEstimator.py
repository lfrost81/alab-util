from alab.train.nn import DNNEstimator
from alab.data import NumpyDataFeeder
from unittest import TestCase


class TestDNNEstimator(TestCase):
    def setUp(self):
        self.conf = {
            'epoch': 1000,
            'learning_rate': 0.05,
            'cost_func': 'cross_entropy',
            'softmax': True,
            'optimizer': 'adam',
            'l2reg_lambda': 0,
            'initializer': 'xavier',
            'hidden_layers': [
                {
                    'dim': 100,
                    'activate_func': 'relu',
                    'dropout': 0.99
                }, {
                    'dim': 33,
                    'activate_func': 'sigmoid',
                    'dropout': 0.99
                },
            ]
        }

        # Input of XOR Problem(0^0=0, 0^1=1, 1^0=1, 1^1=0)
        self.x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.y = [[1, 0], [0, 1], [0, 1], [1, 0]]

    def test_fit(self):
        """
        Test whether model solve XOR classification
        :return: None
        """
        train = DNNEstimator(self.conf, verbosity=2)
        train.fit(self.x, self.y)
        accuracy = train.evaluate(self.x, self.y)
        self.assertEqual(int(accuracy), 1)
        print('Accuracy:', accuracy)
        print('Prediction Result:', train.predict(self.x))

    def test_fit_batch(self):
        """
        Test whether model solve XOR classification on mini batch mode
        :return: None
        """
        feeder = NumpyDataFeeder(fetch_size=4)
        feeder.add_data('x', self.x)
        feeder.add_data('y', self.y)

        train = DNNEstimator(self.conf, verbosity=2)

        train.fit_batch(feeder)
        accuracy = train.evaluate(self.x, self.y)
        self.assertEqual(int(accuracy), 1)
        print('Accuracy:', accuracy)
        print('Prediction Result:', train.predict(self.x))


