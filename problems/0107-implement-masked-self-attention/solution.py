import numpy as np
import math

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""

	def softmax(z):
		maxes = np.max(z, axis=-1, keepdims=True)
		scores = np.exp(z - maxes)
		sums = np.sum(scores, axis=-1, keepdims=True)
		return scores / sums

	_, d = Q.shape

	scores = Q @ K.T * (1.0 / math.sqrt(d)) + mask

	attn = softmax(scores)

	return attn @ V