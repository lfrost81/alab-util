class ALabError(Exception):
    def __init__(self, msg=''):
        self.msg = '%s: %s' % (self.__class__.__name__, msg)

    def __str__(self):
        return repr(self.msg)


class UnimplementedMethodError(ALabError):
    pass
