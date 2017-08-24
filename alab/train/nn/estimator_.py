from alab.exception import UnimplementedMethodError
from alab.data.data_feeder_ import *
from sklearn import metrics

import tensorflow as tf


class NNEstimator:
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

        self.y_hat = None
        self.x_in = None
        self.y_in = None
        self.sess = None

    def draw_graph(self, xdim, ydim):
        raise UnimplementedMethodError()

    def fit(self, x, y):
        raise UnimplementedMethodError()

    def fit_batch(self, feeder: DataFeeder, cut_off=True):
        raise UnimplementedMethodError()

    def predict(self, x, argmax=True):
        predictions = self.sess.run(self.y_hat, feed_dict={self.x_in: x})
        if argmax:
            return np.array([np.argmax(prediction) for prediction in predictions])
        else:
            return np.array([np.argmin(prediction) for prediction in predictions])

    def evaluate(self, x, y, argmax=True):
        labels = self.predict(x, argmax)
        y_true = [np.argmax(a) for a in y]
        accuracy = metrics.accuracy_score(y_true, labels)
        return accuracy


class DNNEstimator(NNEstimator):
    def __init__(self, conf: dict, verbosity=0):
        NNEstimator.__init__(self, verbosity=verbosity)

        # Network configuration
        self.conf = conf

        # Optimizer
        self.optimizer = None

        # Variables
        self.x_in = None
        self.y_in = None
        self.y_hat = None
        self.sess = None
        self.weights = None
        self.biases = None
        self.train = None
        self.cost = None

    def draw_graph(self, xdim, ydim):
        dims = [xdim]
        for layer in self.conf['hidden_layers']:
            dims.append(layer['dim'])
        dims.append(ydim)

        self.x_in = tf.placeholder(tf.float64, [None, xdim])
        self.y_in = tf.placeholder(tf.float64, [None, ydim])

        # Initialize weights and vector for given initializer
        weights = []
        biases = []
        for i in np.arange(len(dims) - 1):
            if self.conf['initializer'] == 'xavier':
                weights.append(tf.Variable(np.random.randn(dims[i],
                                                           dims[i + 1]) / np.sqrt(dims[i])))
            elif self.initializer == 'he':
                weights.append(tf.Variable(np.random.randn(dims[i],
                                                           dims[i + 1]) / np.sqrt(dims[i] / 2)))
            elif self.conf['initializer'] == 'uniform':
                weights.append(tf.Variable(np.random.uniform(-1, 1, [dims[i], dims[i + 1]])))
            else:
                raise
            biases.append(tf.Variable(np.random.normal(-1, 1, [dims[i + 1]])))

        # Set activate functions for the nodes at each layer
        hidden_layers = [self.x_in]
        for i, layer in enumerate(self.conf['hidden_layers']):
            weighted_sum = tf.add(tf.matmul(hidden_layers[-1], weights[i]), biases[i])
            if layer['activate_func'] == 'relu':
                node = tf.nn.relu(weighted_sum)
            elif layer['activate_func'] == 'tanh':
                node = tf.nn.tanh(weighted_sum)
            elif layer['activate_func'] == 'sigmoid':
                node = tf.nn.sigmoid(weighted_sum)
            else:
                raise

            # Set dropouts
            if 'dropout' in layer:
                node = tf.nn.dropout(node, layer['dropout'])

            hidden_layers.append(node)

        # Set output type and cost function for last layer
        self.y_hat = tf.add(tf.matmul(hidden_layers[-1], weights[-1]), biases[-1])
        if self.conf['softmax']:
            self.y_hat = tf.nn.softmax(self.y_hat)

        if self.conf['cost_func'] == 'cross_entropy':
            self.cost = tf.reduce_mean(-tf.reduce_sum(self.y_in * tf.log(self.y_hat),
                                                      reduction_indices=[1]))
        elif self.conf['cost_func'] == 'mse':
            self.cost = tf.reduce_mean(tf.square(self.y_in - self.y_hat))
        else:
            raise

        # Set L2 Regularization
        if self.conf['l2reg_lambda'] > 0:
            for weight in weights:
                self.cost += self.l2reg * tf.nn.l2_loss(weight)

        # Selective optimization
        learning_rate = self.conf['learning_rate']
        if self.conf['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.conf['optimizer'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif self.conf['optimizer'] == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        else:
            raise

        # Set Optimizer
        self.train = self.optimizer.minimize(self.cost)

        self.weights = weights
        self.biases = biases

        # run session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, x, y):
        if type(x) is not np.ndarray:
            x = np.array(x)
        if type(y) is not np.ndarray:
            y = np.array(y)

        self.draw_graph(x.shape[1], y.shape[1])
        for e in np.arange(self.conf['epoch']):
            _, c = self.sess.run([self.train, self.cost], feed_dict={self.x_in: x, self.y_in: y})
            if self.verbosity > 0:
                print('epoch: %s, cost: %s' % (e+1, c))

    def fit_batch(self, feeder: DataFeeder, cut_off=True):
        self.draw_graph(feeder.dim('x'), feeder.dim('y'))
        if self.verbosity > 0:
            print('epoch: %s, cost: %s' % (0, np.inf))
        for e in np.arange(self.conf['epoch']):
            while True:
                x = feeder.feed('x')
                if x is None:
                    break
                if cut_off and len(x) < feeder.fetch_size:
                    break
                y = feeder.feed('y')
                _, c = self.sess.run([self.train, self.cost], feed_dict={self.x_in: x, self.y_in: y})
                if self.verbosity > 1:
                    print('\tmini batch: cost: %s' % (c))
            feeder.rewind()
            if self.verbosity > 0:
                print('epoch: %s, cost: %s' % (e+1, c))

    def get_weights_and_biases(self):
        return self.sess.run([self.weights, self.biases])


class CNNEstimator(NNEstimator):
    def __init__(self, verbosity=0):

        NNEstimator.__init__(self, verbosity=verbosity)


