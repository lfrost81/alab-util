from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove

import itertools
import pprint


def main():
    docs = list(itertools.islice(Text8Corpus('text8'), None))

    ''' Make model '''
    corpus = Corpus()
    corpus.fit(docs, window=10)

    ''' Load Model '''
    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)

    glove.add_dictionary(corpus.dictionary)

    print('man')
    pprint.pprint(glove.most_similar('man', number=10))
    print('flog')
    pprint.pprint(glove.most_similar('flog', number=10))

    return


if __name__ == '__main__':
    main()
