import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	# Your code here	

	mx = np.asarray(input_matrix)

	h, w = mx.shape

	reshaped = mx.reshape(h // pool_size, pool_size, w // pool_size, pool_size)

	return np.mean(reshaped, axis=(1,3))
