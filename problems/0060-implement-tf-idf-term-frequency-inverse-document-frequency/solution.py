import numpy as np
from collections import Counter
import math

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""

	doc_counter = [Counter(d) for d in corpus]
	doc_len = [len(d) for d in corpus]
	
	all_words = [w for d in corpus for w in d]

	counter = Counter(all_words)

	n = len(corpus)

	def tf(w, i):
		return doc_counter[i].get(w, 0) / doc_len[i]

	def df(w):
		return counter.get(w, 0)
	
	def idf(w):
		return math.log((n + 1) / (df(w) + 1)) + 1

	def tfidf(w, i):
		return tf(w, i) * idf(w)

	return [[round(tfidf(w, i), 5) for w in query] for i in range(n)]

