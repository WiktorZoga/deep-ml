import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	pred = np.matmul(X, w)

	mse = np.mean((pred - y_true) ** 2)

	reg = alpha * np.sum(w **2)

	return mse + reg
