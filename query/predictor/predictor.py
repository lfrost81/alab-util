from query.predictor.indexer import GloveLearner
from query.predictor.indexer import LDALearner
from query.predictor.seracher import Searcher
from query.predictor.seracher import DummyMA
from konlpy.tag import Twitter

import numpy as np
import pprint
import pickle
import math


class Predictor:
    def __init__(self, memory_strength):
        self.ms = memory_strength
        self.query_log = set()
        return

    @staticmethod
    def roulette(tuples):
        total = sum([tup[1] for tup in tuples])
        selection = np.random.uniform(0, total)
        accumulated = 0
        for i, (_, score) in enumerate(tuples):
            if accumulated < selection < accumulated + score:
                break
            accumulated += score

        return tuples[i]

    def amnesia(self, query):
        return

    def amnesia_bot(self, query, n_state=10):
        sim_words = self.amnesia(query)

        queries = []
        for _ in range(n_state):
            while True:
                query = Predictor.roulette(sim_words)[0]
                if query not in self.query_log:
                    break

            sim_words = self.amnesia(query)
            queries.append(query)

        return queries

    def ebbinghaus(self, query):
        return

    def ebbinghaus_bot(self, seed_queries, n_state=10):
        for query in seed_queries:
            sim_words = self.ebbinghaus(query)

        queries = []
        for _ in range(n_state):
            while True:
                query = Predictor.roulette(sim_words)[0]
                if query not in self.query_log:
                    break

            sim_words = self.ebbinghaus(query)
            queries.append(query)

        return queries


class SearchBasedPredictor(Predictor):
    def __init__(self, memory_strength, use_ma=False):
        Predictor.__init__(self, memory_strength=memory_strength)
        self.interested_words = {}
        self.sc = Searcher(use_ma=use_ma)

    def amnesia(self, query):
        self.query_log.add(query)

        sim_words = self.sc.find_similar_words(query, filter_queries=self.query_log)
        return sorted(sim_words, key=lambda tup: -tup[1])

    def ebbinghaus(self, query):
        self.query_log.add(query)

        # Diminishing interested_words
        for word, score in self.interested_words.items():
            self.interested_words[word] = score * math.exp(-1 / self.ms)

        # Query
        sim_words = self.sc.find_similar_words(query, filter_queries=self.query_log)
        for word, score in sim_words:
            if word not in self.interested_words:
                self.interested_words[word] = 0
            self.interested_words[word] += score

        result = list(self.interested_words.items())
        return sorted(result, key=lambda tup: -tup[1])

    def clear_interests(self):
        self.interested_words = {}
        self.query_log.clear()


class GloveBasedPredictor(Predictor):
    def __init__(self, memory_strength, use_ma=False):
        Predictor.__init__(self, memory_strength=memory_strength)
        if use_ma:
            self.ma = Twitter()
        else:
            self.ma = DummyMA()

        self.interest_vector = None

        with open(GloveLearner.GLOVE_MODEL_FILE, 'rb') as fp:
            self.glove = pickle.load(fp)

    def get_word_vector_and_queries(self, query):
        vectors = []
        queries = self.ma.nouns(query)
        for word in queries:
            if word not in self.glove.dictionary:
                return None

            word_idx = self.glove.dictionary[query]
            vectors.append(self.glove.word_vectors[word_idx])

        return np.array(vectors).mean(axis=0), queries

    def amnesia(self, query):
        self.query_log.add(query)

        try:
            v, qs = self.get_word_vector_and_queries(query)
        except Exception as e:
            print(e)
            return []

        return self.glove.most_similar_by_vec(v, 10, qs + list(self.query_log))

    def ebbinghaus(self, query):
        self.query_log.add(query)

        try:
            v, qs = self.get_word_vector_and_queries(query)
        except Exception as e:
            print(e)
            return []

        if self.interest_vector is None:
            self.interest_vector = v
        else:
            recent_ratio = math.exp(1 / self.ms)
            self.interest_vector = (self.interest_vector + v * recent_ratio) / (1 + recent_ratio)
            # self.interest_vector = (self.interest_vector + v) / 2

        return self.glove.most_similar_by_vec(self.interest_vector, 10, qs + list(self.query_log))

    def clear_interests(self):
        self.interest_vector = None
        self.query_log.clear()


#TODO: verifying or remove
class LdaBasedPredictor(Predictor):
    def __init__(self, memory_strength, use_ma=False):
        Predictor.__init__(self, memory_strength=memory_strength)
        if use_ma:
            self.ma = Twitter()
        else:
            self.ma = DummyMA()

        self.interest_vector = None

        with open(LDALearner.LDA_MODEL_FILE, 'rb') as fp:
            self.lda = pickle.load(fp)

    def amnesia(self, query):
        term_id = self.lda.id2word.token2id[query]
        print(term_id)
        print(self.lda.get_term_topics(0))


def main():
    topn = 7
    gbp = GloveBasedPredictor(memory_strength=5)
    sbp = SearchBasedPredictor(memory_strength=2)
    qs = ['아인슈타인', '뉴턴', '행성']
    print('Given Queries:', qs)
    print('  Glove Based Prediction(Memorable):')
    for q in qs:
        result = gbp.ebbinghaus(q)[:topn]
        buf = ['(%s: %.3f)' % (word, score) for word, score in result]
        print('    %s => %s' % (q, ', '.join(buf)))
    gbp.clear_interests()
    print('  Glove Based Prediction(Amnesia):')
    for q in qs:
        result = gbp.amnesia(q)[:topn]
        buf = ['(%s: %.3f)' % (word, score) for word, score in result]
        print('    %s => %s' % (q, ', '.join(buf)))
    gbp.clear_interests()

    print('  Search Based Prediction(Memorable):')
    for q in qs:
        result = sbp.ebbinghaus(q)[:topn]
        buf = ['(%s: %.3f)' % (word, score) for word, score in result]
        print('    %s => %s' % (q, ', '.join(buf)))
    sbp.clear_interests()
    print('  Search Based Prediction(Amnesia):')
    for q in qs:
        result = sbp.amnesia(q)[:topn]
        buf = ['(%s: %.3f)' % (word, score) for word, score in result]
        print('    %s => %s' % (q, ', '.join(buf)))
    sbp.clear_interests()

    qs = ['이명박', '노무현', '김대중']
    print()
    print('Given Queries:', qs)
    print('  Glove Based Talker:')
    for _ in range(3):
        print('    =>', gbp.ebbinghaus_bot(qs, n_state=10))
        gbp.clear_interests()

    print('  Search Based Talker:')
    for _ in range(3):
        print('    =>', sbp.ebbinghaus_bot(qs, n_state=10))
        sbp.clear_interests()

    qs = ['이명박', '노무현', '김대중', '신라', '김유신']
    print()
    print('Given Queries:', qs)
    print('  Glove Based Talker:')
    for _ in range(3):
        print('    =>', gbp.ebbinghaus_bot(qs, n_state=10))
        gbp.clear_interests()

    print('  Search Based Talker:')
    for _ in range(3):
        print('    =>', sbp.ebbinghaus_bot(qs, n_state=10))
        sbp.clear_interests()

    print('\n')
    bp = gbp
    prompt = "glove-memorable-diy> "
    while True:
        q = input(prompt)
        if q == '!clear':
            bp.clear_interests()
            continue

        if q == '!change':
            bp.clear_interests()
            if type(bp) is GloveBasedPredictor:
                bp = sbp
                prompt = "search-memorable-diy> "
            else:
                bp = gbp
                prompt = "glove-memorable-diy> "
            continue

        result = bp.ebbinghaus(q)[:topn]
        buf = ['(%s: %.3f)' % (word, score) for word, score in result]
        print('  %s => %s' % (q, ', '.join(buf)))

if __name__ == '__main__':
    main()
