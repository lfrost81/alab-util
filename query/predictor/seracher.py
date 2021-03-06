from query.predictor.indexer import IndexResource
from konlpy.tag import Twitter
import pprint


class DummyMA:
    def nouns(self, query):
        return [query]


class Searcher:
    def __init__(self, use_ma=False):
        self.ir = IndexResource()
        if use_ma:
            self.ma = Twitter()
        else:
            self.ma = DummyMA()

    def get_posting(self, query):
        docids = set()
        for word in self.ma.nouns(query):
            if word in self.ir.inverted_index:
                if len(docids) == 0:
                    docids = set(self.ir.inverted_index[word])
                else:
                    docids = docids.intersection(set(self.ir.inverted_index[word]))

        return list(docids)

    def search(self, query, n_doc=10):
        docids = self.get_posting(query)
        docs = [self.ir.read_doc(docid) for docid in docids]
        return sorted(docs, key=lambda doc: -doc[query])[0:n_doc]

    def __collect_words_in_doc(self, query, n_doc=300):
        docs = self.search(query, n_doc)
        collected = dict()
        for doc in docs:
            for word, freq in doc.items():
                if word not in collected:
                    collected[word] = 0
                collected[word] += freq

        return collected

    def find_significant_words(self, query, n_doc=300):
        collected = self.__collect_words_in_doc(query, n_doc)
        normalized_collected = []
        for word, freq in collected.items():
            idf = self.ir.get_idf(word)
            normalized_collected.append((word, freq / idf))

        return sorted(normalized_collected, key=lambda tup: -tup[1])

    def find_similar_words(self, query, n_candidate=1000, n_doc=300, n_word=100, filter_queries=[],
                           metric='jaccard'):
        query_feature = set(self.get_posting(query))
        query_set = set(self.ma.nouns(query))
        similar_words = []

        for i, (candidate, _) in enumerate(self.find_significant_words(query, n_doc)):
            if candidate in query_set or candidate in filter_queries:
                continue

            posting = self.get_posting(candidate)
            if metric == 'jaccard':
                intersected = query_feature.intersection(posting)
                union = query_feature.union(posting)
                word_n_score = (candidate, len(intersected) / len(union))
            else:
                intersected = query_feature.intersection(posting)
                dist = ((len(query_feature) * len(posting)) / (len(intersected) ** 2)) ** 0.5
                score = 1 / dist
                word_n_score = (candidate, score)

            similar_words.append(word_n_score)

            if i > n_candidate:
                break

        similar_words = sorted(similar_words, key=lambda tup: -tup[1])

        if len(similar_words) > n_word:
            similar_words = similar_words[0:n_word]

        return similar_words


if __name__ == '__main__':
    sc = Searcher()
    q = '대통령'
    print(sc.search(q))
    pprint.pprint(sc.find_similar_words(q))

    # pprint.pprint(sc.find_similar_words('아인슈타인'))
    # pprint.pprint(sc.find_similar_words('뉴턴'))
    # pprint.pprint(sc.find_similar_words('블랙홀'))
    #print(sc.search('아인슈타인'))
    #print(sc.ir.get_idf('아인슈타인'))
