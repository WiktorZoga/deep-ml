import numpy as np
from collections import Counter
import math

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):

	counts = [Counter(d) for d in corpus]

	N = len(corpus)

	def f(t, i):
		return counts[i].get(t, 0)

	def n(t):
		return sum([f(t, i) > 0 for i in range(N)])

	avgdl = sum([len(d) for d in corpus]) / N

	def score(i):
		s = 0
		for t in query:
			ff = f(t, i)
			s += math.log((N + 1) / (n(t) + 1)) * ff * (k1 + 1) / (ff + k1 * (1 - b + b*len(corpus[i])/avgdl))
		return  s

	scores = [score(i) for i in range(N)]

	return np.round(scores,3)