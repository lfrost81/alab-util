import numpy as np
from alab.exception import UnimplementedMethodError


class FeedingElement:
    def __init__(self, data):
        self.proceed_row_indice = -1
        self.data = data
        self.max_row = self.row_len()

    def feed_data(self, fetch_size):
        raise UnimplementedMethodError()

    def row_len(self):
        raise UnimplementedMethodError()

    def col_len(self):
        raise UnimplementedMethodError()

    def rewind(self):
        self.proceed_row_indice = -1


class NumpyFeedingElement(FeedingElement):
    def __init__(self, data):
        FeedingElement.__init__(self, data)

    def feed_data(self, fetch_size):
        row_from = self.proceed_row_indice + 1
        if self.max_row > row_from + fetch_size:
            row_end = row_from + fetch_size
        else:
            row_end = self.max_row

        self.proceed_row_indice = row_end - 1

        return self.data[row_from:row_end]

    def row_len(self):
        return len(self.data)

    def col_len(self):
        return len(self.data[0])


class DataFeeder:
    def __init__(self, fetch_size):
        self.fetch_size = fetch_size
        self.last_col_index = -1
        self.elements = {}

    def add_data(self, var_name, data):
        raise UnimplementedMethodError()

    def feed(self, var_name):
        raise UnimplementedMethodError()

    def dim(self, var):
        return self.elements[var].col_len()

    def rewind(self):
        for key, val in self.elements.items():
            val.rewind()


class NumpyDataFeeder(DataFeeder):
    def __init__(self, fetch_size=1):
        DataFeeder.__init__(self, fetch_size)

    def add_data(self, var_name, data):
        if type(data) is not np.ndarray:
            data = np.array(data)

        element = NumpyFeedingElement(data)
        self.elements[var_name] = element

    def feed(self, var_name):
        result = self.elements[var_name].feed_data(self.fetch_size)
        if len(result) <= 0:
            return None
        return result


