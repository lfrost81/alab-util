import time


class SimpleProfiler:
    def __init__(self):
        self.begin = time.time()
        self.checkpoint = self.begin
        self.lap = 0
        return

    def laps(self):
        cur = time.time()
        self.lap += 1
        print('Lap(%s) Elapsed: %.4f sec, Total Elapsed: %.4f sec' %
              (self.lap, cur - self.checkpoint, cur - self.begin))
        self.checkpoint = cur
        return

