# imports
from inverted_index_gcp import *
from collections import Counter, defaultdict
from nltk.stem.porter import *
import numpy as np
import hashlib
from contextlib import closing
from nltk.corpus import stopwords


# preprocessing

class Preprocess:

    def __init__(self):
        self.NUM_BUCKETS = 124
        self.TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        self.TUPLE_SIZE = 6
        self.re_word = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.docs_len = defaultdict(int)  # doc lens
        self.super_posting_locs = defaultdict(list)
        self.tf = defaultdict(list)
        self.idf = defaultdict(int)
        self.cos_sin = defaultdict(list)
        self.query = []
        self.title_search = defaultdict(int)

    def tokenizer(self, query):
        english_stopwords = frozenset(stopwords.words('english'))
        toked_q = query.lower().split()
        new_query = []
        for token in toked_q:
            if token in english_stopwords:
                continue
            new_query.append(token)
        new_query = ' '.join(new_query)
        return new_query

    def remove_stopwords(self, query):
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        all_stopwords = english_stopwords.union(corpus_stopwords)
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        # YOUR CODE HERE
        tokens_clean = [token for token in tokens if token not in all_stopwords]
        self.query = tokens_clean

    # document length
    def clac_d_lens(self, docs):
        docs = docs.collect()
        for doc in range(len(docs)):
            self.docs_len[list(docs)[doc][0]] = len(list(docs)[doc][2].split())

    def word_count(self, id, text):
        """ Count the frequency of each word in `text` (tf) that is not included in
      `all_stopwords` and return entries that will go into our posting lists.
      Parameters:
      -----------
        text: str
          Text of one document
        id: int
          Document id
      Returns:
      --------
        List of tuples
          A list of (token, (doc_id, tf)) pairs
          for example: [("Anarchism", (12, 5)), ...]
        """
        tokens_text = [token.group() for token in self.re_word.finditer(text.lower())]
        tokens = tokens_text
        # YOUR CODE HERE
        res = []
        counter = Counter(tokens)
        for key, val in counter.items():
            res.append((key, (id, val)))
        return res

    @staticmethod
    def reduce_word_counts(unsorted_pl):
        """ Returns a sorted posting list by wiki_id.
      Parameters:
      -----------
        unsorted_pl: list of tuples
          A list of (wiki_id, tf) tuples
      Returns:
      --------
        list of tuples
          A sorted posting list.
        """
        # YOUR CODE HERE
        res = sorted(unsorted_pl)
        return res

    @staticmethod
    def calculate_df(postings):
        """ Takes a posting list RDD and calculate the df for each token.
        Parameters:
        -----------
          postings: RDD
            An RDD where each element is a (token, posting_list) pair.
        Returns:
        --------
          RDD
            An RDD where each element is a (token, df) pair.
            """
        # YOUR CODE HERE
        res = postings.mapValues(len)
        return res

    @staticmethod
    def hash(s):
        return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

    def token2bucket_id(self, token):
        return int(self.hash(s=token), 16) % self.NUM_BUCKETS

    def partition_postings_and_write(self, postings):
        """ A function that partitions the posting lists into buckets, writes out
        all posting lists in a bucket to disk, and returns the posting locations for
        each bucket. Partitioning should be done through the use of `token2bucket`
        above. Writing to disk should use the function  `write_a_posting_list`, a
        static method implemented in inverted_index_colab.py under the InvertedIndex
        class.
        Parameters:
        -----------
          postings: RDD
            An RDD where each item is a (w, posting_list) pair.
        Returns:
        --------
          RDD
            An RDD where each item is a posting locations dictionary for a bucket. The
            posting locations maintain a list for each word of file locations and
            offsets its posting list was written to. See `write_a_posting_list` for
            more details.
            """
        # YOUR CODE HERE
        calc_bucket_id = postings.map(lambda word: (self.token2bucket_id(word[0]), word))
        res = calc_bucket_id.groupByKey().map(lambda entry: InvertedIndex.write_a_posting_list(entry))
        return res

    def super_p_loc(self, posting_locs_list):
        for posting_loc in posting_locs_list:
            for k, v in posting_loc.items():
                self.super_posting_locs[k].extend(v)

    def read_posting_list(self, inverted, w):
        with closing(MultiFileReader()) as reader:
            locs = inverted.posting_locs[w]
            for i, tup in enumerate(locs):
                locs[i] = tuple(list([tup[0], tup[1]]))

            b = reader.read(locs, inverted.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def tf_idf(self, inverted, N):
        for word in self.query:
            try:
                temp = self.read_posting_list(inverted, word)
                df = len(temp)
                self.idf[word] = np.log2((N + 1) / (df + 1))
                dict_tf = defaultdict(int)
                for tup in temp:
                    dict_tf[tup[0]] = tup[1] / inverted.d_len[tup[0]]
                self.tf[word] = dict_tf.items()
            except:
                continue

    def tf_try(self, pl, query, docs):
        for word in query.split():
            temp = pl[word]
            df = len(temp)
            self.idf[word] = np.log2((4 / df))
            dict_tf = defaultdict(int)
            for tup in temp:
                dict_tf[tup[0]] = tup[1] / docs[tup[0] - 1]
            self.tf[word] = dict_tf.items()

    def calc_tfIdf(self, query, inverted):
        query_len = len(self.query)
        tfIDF = defaultdict(int)
        d_lens = defaultdict(int)
        for word in self.idf:
            for tup in self.tf[word]:
                tfIDF[tup[0]] += tup[1] * self.idf[word]
                d_lens[tup[0]] = inverted.d_len[tup[0]]
        for d in d_lens:
            tfIDF[d] = tfIDF[d] / (d_lens[d] * query_len)
        self.cos_sin[query] = sorted(tfIDF.items(), key=lambda item: item[1], reverse=True)

    def try_tfidf(self, query, docs):
        query_len = 2
        tfIDF = defaultdict(int)
        d_lens = defaultdict(int)
        for word in self.idf:
            for tup in self.tf[word]:
                tfIDF[tup[0]] += tup[1] * self.idf[word]
                d_lens[tup[0]] = docs[tup[0] - 1]
        for d in d_lens:
            tfIDF[d] = tfIDF[d] / (d_lens[d] * 2)
        self.cos_sin[query] = sorted(tfIDF.items(), key=lambda item: item[1], reverse=True)
   
