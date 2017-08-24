import numpy as np
import re

class HiveDataTransform:
    context = ''
    col_names = 'name, freq, amt, weight'

    #todo: delete
    test_data = [
        ['권영현 짠', 10, 20, 0.8],
        ['박영서 샤샤', 30, 20, 0.2],
        ['주신홍 구구', 40, 5, 0.1],
        ['이석민 추추', 342, 13, 11.23],
        ['남궁현 키키', 324, 1, 13, 32.2]
    ]
    cur_row_idx = -1
    test_data_len = len(test_data)

    def __init__(self, context=''):
        self.context = context
        return

    #Todo: change to sql cursor
    def __next(self, cols=None):
        self.cur_row_idx += 1
        if self.test_data_len <= self.cur_row_idx:
            return None

        row = self.test_data[self.cur_row_idx]

        if cols is None:
            return row

        new_row = []
        for col in re.split(',', cols):
            new_row.append(row[int(col)])

        return new_row

    def rewind(self):
        self.cur_row_idx = -1
        return

    def to_matrix(self, cols=None):
        mat = []
        while True:
            row = self.__next(cols)
            if row is None:
                break
            mat.append(row)
        mat = np.array(mat)
        return mat

    def to_table(self, cols=None):
        col_names = re.split(',', self.col_names)
        col_name_dict = {}
        for i, col_name in enumerate(col_names):
            col_name_dict[col_name.strip()] = i
        return col_name_dict, self.to_matrix(cols)

    def to_raw_transaction(self, key_col, columns=None):

        return







    def __to_flat_list(self, columns=None):
        lst = []
        while True:
            row = self.__next(columns)
            if row is None:
                break
            for col in row:
                lst.append(col)
        return lst

    def __to_word_list(self, columns=None):
        lst = []
        while True:
            row = self.__next(columns)
            if row is None:
                break
            for col in row:
                for word in re.split(' ', col):
                    lst.append(word)
        return lst
