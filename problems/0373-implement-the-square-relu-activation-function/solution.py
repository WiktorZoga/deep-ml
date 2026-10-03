import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""

	def f(x):
		return np.maximum(0.0, x) ** 2.0
	
	def df(x):
		return 2.0 * np.maximum(0.0, x)
	
	return {
		"output": f(x).round(4),
		"derivative": df(x).round(4),
	}