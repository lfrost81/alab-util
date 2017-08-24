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


# class Feeder:
#     def __init__(self, schema):
#         self.schema = schema
#         if schema is None or len(self.schema) == 0:
#             self.schema = {'*': np.inf}
#         self.row_index = 0
#         if type(self.schema) is not dict:
#             raise
#
#         return
#
#     def feed(self):
#         raise UnimplementedMethodError()
#
#     def rewind(self):
#         self.row_index = 0
#
#
# class BatchFeeder(Feeder):
#     def __init__(self, feeder: Feeder, fetch_size):
#         Feeder.__init__(self, None)
#         self.feeder = feeder
#         self.fetch_size = fetch_size
#         return
#
#     def feed(self):
#         results = {}
#         for i in range(self.fetch_size):
#             row = self.feeder.feed()
#             if row is None:
#                 if i == 0:
#                     return None
#                 break
#
#             for var in row:
#                 if i == 0:
#                     results[var] = [row[var]]
#                 else:
#                     results[var].append(row[var])
#
#             self.row_index += 1
#
#         for var in results:
#             results[var] = np.array(results[var])
#
#         return results
#
#
# class NumpyFeeder(Feeder):
#     def __init__(self, array: np.ndarray, schema=None):
#         Feeder.__init__(self, schema)
#         self.array = array
#         self.max_index = array.shape[0]
#         return
#
#     def feed(self):
#         results = {}
#         for var in self.schema:
#             if self.row_index < self.max_index:
#                 if var == '*':
#                     results[var] = self.array[self.row_index, :]
#                 else:
#                     results[var] = self.array[self.row_index, self.schema[var]]
#             else:
#                 return None
#
#         self.row_index += 1
#         return results
#
#
# class HiveFeeder(Feeder):
#     def __init__(self, query: str, fetch_size=10000, schema=None):
#         self.fetch_size = fetch_size
#         return
#
#     def feed(self):
#         return
#
#
# class FileFeeder(Feeder):
#     def __init__(self, filename: str, fetch_size=10000, schema=None):
#         self.fetch_size = fetch_size
#         return
#
#     def feed(self):
#         return


# feeder = NumpyFeeder(np.random.uniform(0, 1, [3, 3]), schema={'x': range(0, 3)})
# feeder = BatchFeeder(feeder, fetch_size=2)
#
# while True:
#     result = feeder.feed()
#     if result is None:
#         break
#
#     print(result)

