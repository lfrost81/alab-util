from multiprocessing import Process
from konlpy.tag import Twitter

import xml.etree.ElementTree as ET
import pickle
import pprint
import queue
import time
import os
import re


class Indexer:
    PREPROCESSED_FILE = 'preprocessed.bin'
    INTER_FORWARD_INDEX_FILE = 'forward_index_%09d.bin'
    FORWARD_INDEX_FILE = 'forward_index.bin'
    FORWARD_OFFSET_FILE = 'forward_offset.bin'
    INVERTED_INDEX_FILE = 'inverted_index.bin'

    def __init__(self, src_file=''):
        self.src_file = src_file
        return

    def preprocess(self, max_doc=0):
        docid = 0

        forward_index = []
        with open(self.src_file, 'r', encoding='utf-8') as fp:
            xml_str = ''
            is_page_start = False
            is_page_end = False

            for i, line in enumerate(fp):
                if re.search('<page>', line):
                    is_page_start = True
                if re.search('</page>', line):
                    is_page_end = True

                if is_page_start:
                    xml_str += line
                if is_page_end:
                    root = ET.fromstring(xml_str)
                    title = root.find('title').text
                    text = root.find('revision').find('text').text
                    if title is None or text is None:
                        contents = ''
                    else:
                        contents = title + ' ' + text
                        contents = re.sub('[0-9\W]', ' ', contents)
                        contents = re.sub('\s+', ' ', contents)

                    forward_index.append(contents)
                    is_page_start = False
                    is_page_end = False
                    xml_str = ''

                    docid += 1
                    if docid % 1000 == 0:
                        print(docid)

                    if max_doc == docid:
                        break

        print(docid)
        with open(self.PREPROCESSED_FILE, 'wb') as ofp:
            pickle.dump(forward_index, ofp)

    @staticmethod
    def forward_index_worker(begin=0, size=10000):
        with open('preprocessed.bin', 'rb') as fp:
            forward_index = pickle.load(fp)

        end = begin + size
        if len(forward_index) < end:
            end = len(forward_index)
        forward_index = forward_index[begin:end]

        tma = Twitter()
        doc_count = 0
        for i, doc in enumerate(forward_index):
            docid = i + begin
            if begin > docid:
                continue

            if docid % 1000 == 0:
                print(docid)

            doc = str(bytes(doc, encoding='cp949', errors='replace'), encoding='cp949')
            doc = ' '.join([word for word in doc.split(' ') if len(word) < 20])
            ### tmp_index = dict()
            ### for noun in tma.nouns(doc):
            ###     if noun not in tmp_index:
            ###         tmp_index[noun] = 0
            ###     tmp_index[noun] += 1

            forward_index[i] = tma.nouns(doc)

            doc_count += 1
            if doc_count == size:
                break

        print(docid)
        with open(Indexer.INTER_FORWARD_INDEX_FILE % begin, 'wb') as ofp:
            pickle.dump(forward_index, ofp)

    def merge_forward_index(self):
        files = [path for path in os.listdir('.') if re.match('forward_index_\d+.bin', path)]
        files = sorted(files)
        offsets = [0]
        ofp = open('forward_index.bin', 'wb')
        for file in files:
            with open(file, 'rb') as fp:
                docs = pickle.load(fp)
                for doc in docs:
                    s = pickle.dumps(doc)
                    offsets.append(offsets[-1] + len(s))
                    ofp.write(s)

        ofp.close()
        with open(self.FORWARD_INDEX_FILE, 'wb', buffering=1024*1024) as ofp:
            pickle.dump(offsets, ofp)

    def create_inverted_index(self):
        return

    def run(self):
        self.preprocess()

        processes = []
        index_block = 400000
        for begin in range(0, 1600000, index_block):
            process = Process(target=Indexer.forward_index_worker, args=(begin, index_block))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        self.merge_forward_index()
        self.load_all_forward_index()


class IndexResource:
    FORWARD_INDEX_FILE = Indexer.FORWARD_INDEX_FILE
    FORWARD_OFFSET_FILE = Indexer.FORWARD_OFFSET_FILE
    INVERTED_INDEX_FILE = Indexer.INVERTED_INDEX_FILE

    def __init__(self):
        with open(self.FORWARD_OFFSET_FILE, 'rb') as fp:
            self.offsets = pickle.load(fp)

        self.forward_index = open(self.FORWARD_INDEX_FILE, 'rb')

    def load_all_forward_index(self):
        docs = []
        for i, offset in enumerate(self.offsets):
            size = self.offsets[i+1] - offset
            self.forward_index.seek(offset)
            doc = pickle.loads(self.forward_index.read(size))
            docs.append(doc)

    def get_doc(self, docid):
        offset = self.offsets[docid]
        size = self.offsets[docid+1] - offset
        self.forward_index.seek(offset)
        return pickle.loads(self.forward_index.read(size))

if __name__ == '__main__':
    #indexer = Indexer('D:/My/Documents/data/kowiki-20170820-pages-meta-current.xml')
    #indexer.create_inverted_index()

    ir = IndexResource()
    print(ir.get_doc(49538))

    #indexer.run()



