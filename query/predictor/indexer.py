from gensim.corpora import Dictionary
from multiprocessing import Process
from gensim.models import LdaModel
from glove import Glove, Corpus
from konlpy.tag import Twitter

import xml.etree.ElementTree as ET
import pickle
import os
import re


class Indexer:
    IO_BUFFER = 1024*1024
    DATA_DIR = './data/'
    PREPROCESSED_FILE = DATA_DIR + 'preprocessed.bin'
    INTER_FORWARD_INDEX_FILE = DATA_DIR + 'forward_index_%09d.bin'
    INTER_FORWARD_INDEX_FILE_PATTERN = 'forward_index_\d+.bin'
    FORWARD_INDEX_FILE = DATA_DIR + 'forward_index.bin'
    FORWARD_OFFSET_FILE = DATA_DIR + 'forward_offset.bin'
    INVERTED_INDEX_FILE = DATA_DIR + 'inverted_index.bin'

    def __init__(self, src_file):
        os.makedirs('./data', exist_ok=True)
        self.src_file = src_file
        return

    def preprocess_scd(self):
        docs = []
        doc = ''
        fp = open(self.src_file, 'r', encoding='utf-8')
        for i, line in enumerate(fp):
            if line.startswith('<DOCID>'):
                if len(doc) > 0:
                    doc = re.sub('[0-9\W]', ' ', doc)
                    doc = re.sub('\s+', ' ', doc)
                    docs.append(doc)
                    doc = ''
            else:
                m = re.match('^<.*?>(.*)$', line)
                doc += m.group(1) + ' '

        fp.close()

        with open(self.PREPROCESSED_FILE, 'wb') as ofp:
            pickle.dump(docs, ofp)

    def preprocess_xml(self, max_doc=0):
        docs = []
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

                    docs.append(contents)
                    is_page_start = False
                    is_page_end = False
                    xml_str = ''

                    if i+1 % 1000 == 0:
                        print('Preprocessing:', i)

                    if max_doc == i+1:
                        break

        print('Preprocessing:', i)
        with open(self.PREPROCESSED_FILE, 'wb') as ofp:
            pickle.dump(docs, ofp)

    @staticmethod
    def keyword_extraction_worker(begin=0, size=10000):
        with open(Indexer.PREPROCESSED_FILE, 'rb') as fp:
            docs = pickle.load(fp)

        end = begin + size
        if len(docs) < end:
            end = len(docs)
        docs = docs[begin:end]

        tma = Twitter()

        try:
            os.remove(Indexer.INTER_FORWARD_INDEX_FILE % begin)
        except FileNotFoundError:
            pass

        ofp = open(Indexer.INTER_FORWARD_INDEX_FILE % begin, 'a', buffering=Indexer.IO_BUFFER)
        for i, doc in enumerate(docs):
            doc = str(bytes(doc, encoding='cp949', errors='replace'), encoding='cp949')
            doc = ' '.join([word for word in doc.split(' ') if 1 < len(word) < 20])
            doc = ' '.join(tma.nouns(doc))
            ofp.write(doc + '\n')
            docs[i] = None

            if i+1 % 1000 == 0:
                print('Extracting keyword:', begin + i)

        print('Extracting keyword:', begin + i)
        ofp.close()

    @staticmethod
    def load_keyword_extracted_docs():
        files = [path for path in os.listdir(Indexer.DATA_DIR)
                 if re.search(Indexer.INTER_FORWARD_INDEX_FILE_PATTERN, path)]
        files = sorted(files)
        docs = []
        for file in files:
            with open(Indexer.DATA_DIR + file, 'r') as fp:
                for nouns_str in fp:
                    nouns_str = re.sub('\n+', '', nouns_str)
                    docs.append(nouns_str.split(' '))

        return docs

    def merge_forward_index(self):
        files = [path for path in os.listdir(self.DATA_DIR)
                 if re.search(self.INTER_FORWARD_INDEX_FILE_PATTERN, path)]
        files = sorted(files)
        offsets = [0]
        ofp = open(self.FORWARD_INDEX_FILE, 'wb', buffering=Indexer.IO_BUFFER)
        doc_count = 0
        for file in files:
            with open(self.DATA_DIR + file, 'r') as fp:
                for nouns_str in fp:
                    nouns_str = re.sub('\n+', '', nouns_str)
                    nouns = nouns_str.split(' ')
                    tmp_index = dict()
                    for noun in nouns:
                        if noun not in tmp_index:
                            tmp_index[noun] = 0
                        tmp_index[noun] += 1

                    doc_bytes = pickle.dumps(tmp_index)
                    offsets.append(offsets[-1] + len(doc_bytes))
                    ofp.write(doc_bytes)

                    doc_count += 1
                    if doc_count % 1000 == 0:
                        print('Merging:', doc_count)

        ofp.close()
        print('Merging:', doc_count)
        with open(self.FORWARD_OFFSET_FILE, 'wb', buffering=Indexer.IO_BUFFER) as ofp:
            pickle.dump(offsets, ofp)

    def create_inverted_index(self):
        inverted_index = dict()
        ir = IndexResource(inverted_index=False)
        for docid in range(ir.doc_count()):
            doc = ir.read_doc(docid)
            for word in doc:
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(docid)
            if docid > 10000:
                break

        with open(self.INVERTED_INDEX_FILE, 'wb') as ofp:
            pickle.dump(inverted_index, ofp)

    def run(self, source='xml', max_doc=0, partitions=None):
        print('Starting preprocess...')
        if source == 'xml':
            self.preprocess_xml(max_doc=max_doc)
        elif source == 'scd':
            self.preprocess_scd()
        print('End preprocess.')

        print('Starting keyword extraction...')
        processes = []
        if partitions is None:
            index_block = 400000
            for begin in range(0, 1600000, index_block):
                process = Process(target=Indexer.keyword_extraction_worker, args=(begin, index_block))
                processes.append(process)
        else:
            begin = 0
            for size in partitions:
                process = Process(target=Indexer.keyword_extraction_worker, args=(begin, size))
                processes.append(process)
                begin += size

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        print('End keyword extraction.')

        print('Starting merge...')
        self.merge_forward_index()
        print('End merge.')

        print('Starting index...')
        self.create_inverted_index()
        print('End index.')


class IndexResource:
    def __init__(self, inverted_index=True):
        with open(Indexer.FORWARD_OFFSET_FILE, 'rb') as fp:
            self.offsets = pickle.load(fp)

        self.forward_index = open(Indexer.FORWARD_INDEX_FILE, 'rb')
        self.inverted_index = None
        if inverted_index:
            with open(Indexer.INVERTED_INDEX_FILE, 'rb') as fp:
                self.inverted_index = pickle.load(fp)

    def load_all_docs(self):
        docs = []
        for i, offset in enumerate(self.offsets):
            size = self.offsets[i+1] - offset
            self.forward_index.seek(offset)
            doc = pickle.loads(self.forward_index.read(size))
            docs.append(doc)

    def read_doc(self, docid):
        offset = self.offsets[docid]
        size = self.offsets[docid+1] - offset
        self.forward_index.seek(offset)
        return pickle.loads(self.forward_index.read(size))

    def doc_count(self):
        return len(self.offsets) - 1

    def search(self, query, topn=10):
        if query in self.inverted_index:
            return [self.read_doc(docid) for i, docid in enumerate(self.inverted_index[query])
                    if i < topn]
        else:
            return []

    def get_idf(self, query):
        if query in self.inverted_index:
            return len(self.inverted_index[query])
        return 0


class GloveLearner:
    GLOVE_MODEL_FILE = Indexer.DATA_DIR + 'glove.bin'

    def __init__(self, n_component, epochs, window_size=10, thread_no=3):
        self.n_component = n_component
        self.window_size = window_size
        self.epochs = epochs
        self.thread_no = thread_no

    def fit(self):
        docs = Indexer.load_keyword_extracted_docs()

        corpus = Corpus()
        corpus.fit(docs, window=self.window_size)

        glove = Glove(self.n_component)
        glove.fit(corpus.matrix, epochs=self.epochs, no_threads=self.thread_no, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        with open(self.GLOVE_MODEL_FILE, 'wb') as ofp:
            pickle.dump(glove, ofp)


class LDALearner:
    LDA_MODEL_FILE = Indexer.DATA_DIR + 'lda.bin'

    def __init__(self, n_topic):
        self.n_topic = n_topic

    def fit(self):
        docs = Indexer.load_keyword_extracted_docs()
        dictionary = Dictionary(docs)
        for i, doc in enumerate(docs):
            docs[i] = dictionary.doc2bow(doc)

        lda = LdaModel(docs, num_topics=self.n_topic, id2word=dictionary)
        with open(self.LDA_MODEL_FILE, 'wb') as ofp:
            pickle.dump(lda, ofp)


def main():
    #indexer = Indexer('D:/My/Documents/data/B-00-201505122213-17936-U-C.SCD')
    #indexer.run()

    #gl = GloveLearner(200, 30, window_size=10, thread_no=9)
    #gl.fit()

    ll = LDALearner(100)
    ll.fit()


if __name__ == '__main__':
    main()

