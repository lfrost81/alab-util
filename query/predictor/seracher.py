from query.predictor.indexer import IndexResource
from konlpy.tag import Twitter


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
        return [self.ir.read_doc(docid) for i, docid in enumerate(docids) if i < n_doc]

    def find_significant_words(self, query, n_doc=1000):
        docs = self.search(query, n_doc)
        collected = dict()
        for doc in docs:
            for word, freq in doc.items():
                if word not in collected:
                    collected[word] = 0
                collected[word] += freq

        normalized_collected = []
        for word, freq in collected.items():
            idf = self.ir.get_idf(word)
            normalized_collected.append((word, freq / idf))

        return sorted(normalized_collected, key=lambda tup: -tup[1])

    def find_similar_words(self, query, n_candidate=10000, n_word=10):
        query_feature = set(self.get_posting(query))
        query_set = set(self.ma.nouns(query))
        similar_words = []
        for i, (candidate, _) in enumerate(self.find_significant_words(query)):
            if candidate in query_set:
                continue

            posting = self.get_posting(candidate)
            intersected = query_feature.intersection(posting)
            union = query_feature.union(posting)
            similar_words.append((candidate, len(intersected) / len(union)))
            if i > n_candidate:
                break

        similar_words = sorted(similar_words, key=lambda tup: -tup[1])
        if len(similar_words) > n_word:
            return similar_words[0:n_word]

        return similar_words


if __name__ == '__main__':
    sc = Searcher()
    q = '아인슈타인 뉴턴'
    print(sc.search(q))
    #print(sc.find_similar_words(q))

    # pprint.pprint(sc.find_similar_words('아인슈타인'))
    # pprint.pprint(sc.find_similar_words('뉴턴'))
    # pprint.pprint(sc.find_similar_words('블랙홀'))
    #print(sc.search('아인슈타인'))
    #print(sc.ir.get_idf('아인슈타인'))
