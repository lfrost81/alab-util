#-*-coding:utf-8 -*-

import numpy as np
import threading
import random
import queue
import json
import re
import os
import path


class TagRandGenerator():
    def __init__(self, conf_path, count=100, output_dir_path=None, date=None,
                 column_generate_prob=.1, threadno=4, print_per=1000):

        with open(conf_path, 'r') as fp:
            self.conf = json.load(fp)
        self.schema = self.conf['schema']
        self.obj_name = self.conf['obj_name']

        self.count = int(count)
        self.dic_path = os.path.join(conf_path, self.obj_name + '.dic.json')
        self.id_tag = self.obj_name
        self.output_path = os.path.join(output_dir_path, self.obj_name + '_' + date[0:6] + '.txt')
        self.date = date
        self.column_generate_prob = column_generate_prob

        self.input_queue = queue.Queue(maxsize=100000)
        self.output_queue = queue.Queue(maxsize=100000)
        self.perm_size = 10000000
        self.threadno = threadno
        self.write_buffer_size = 1000
        self.print_per = print_per

    def process(self):
        threads = []

        print(self.date[0:6] + ' is generating')
        thread = threading.Thread(target=TagRandGenerator.feeder, args=(self,))
        threads.append(thread)
        for i in range(self.threadno):
            thread = threading.Thread(target=TagRandGenerator.generator, args=(self,))
            threads.append(thread)
        thread = threading.Thread(target=TagRandGenerator.writer, args=(self,))
        threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    @staticmethod
    def feeder(self):
        result_dict = {}

        for i in range(self.count):
            identifier = i + 1
            if identifier % self.print_per == 0:
                print('feed:', identifier)

            result_dict['id'] = identifier
            result_dict['classType'] = self.id_tag
            result_dict['date'] = self.date
            self.input_queue.put(result_dict)
            result_dict = {}

        # Mark End
        for i in range(self.threadno):
            self.input_queue.put(None)

        return

    @staticmethod
    def generator(self):
        processed = 0
        while True:
            item = self.input_queue.get()

            # Exit
            if item is None:
                break

            result_dict = item

            # Generate random values
            for k, v in self.schema.items():
                if np.random.rand() > self.conf['occur_prob']:
                    continue

                if type(v) is list:
                    value = np.random.choice(v)
                    if type(value) is np.int64:
                        value = int(value)
                    else:
                        value = str(value)
                else:
                    value = int(np.random.randint(v['gte'], v['lte']))

                result_dict[k] = value

            result = '{}\n'.format(json.dumps(result_dict, ensure_ascii=False))
            self.output_queue.put(result)

            processed += 1
            if processed % self.print_per == 0:
                print('generate', processed)

        self.output_queue.put(None)

        return

    @staticmethod
    def writer(self):
        processed = 0
        done_count = 0
        file = open(self.output_path, 'w')
        max_buf_size = self.write_buffer_size
        buf_size = 0
        buf = []
        while True:
            # Exit
            result = self.output_queue.get()
            if result is None:
                done_count += 1
                if done_count == self.threadno:
                    break
                else:
                    continue

            buf.append(result)
            buf_size += 1
            if buf_size == max_buf_size:
                for result in buf:
                    file.write(result)
                buf_size = 0
                buf = []

            processed += 1
            if processed % self.print_per == 0:
                print('written', processed)

        for result in buf:
            file.write(result)

        file.close()
        return


def main():
    year_start = 2017
    years = 1
    per_months = 1
    for i in range(years):
        date = year_start + i
        date *= 10000
        date += 1
        for j in range(per_months):
            date += 100
            rand_generator = TagRandGenerator('conf/customer.ptag_gen.json',
                                              count=10, date=str(date), threadno=1,
                                              output_dir_path=os.path.join(path.ROOT, 'output'))
            rand_generator.process()


if __name__ == "__main__":
    main()
