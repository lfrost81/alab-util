

class DataSplitter:
    def __init__(self, X, y=None, ratio=0.8):
        self.ratio = ratio
        self.X = X
        self.total_size = len(X)
        self.y = y

        if self.y is not None:
            if self.total_size != len(self.y):
                raise

        self.train_amount = int(self.total_size * self.ratio)

        return

    def split(self):
        if self.y is None:
            return self.X[0:self.train_amount], self.X[self.train_amount:]
        else:
            return (self.X[0:self.train_amount], self.X[self.train_amount:],
                    self.y[0:self.train_amount], self.y[self.train_amount:])
