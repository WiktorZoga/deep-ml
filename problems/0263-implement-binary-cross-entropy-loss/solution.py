import numpy as np

def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here

	y_true = np.clip(np.asarray(y_true), epsilon, 1 - epsilon)
	y_pred = np.clip(np.asarray(y_pred), epsilon, 1 - epsilon)

	bce_loos = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

	return bce_loos.tolist()
	
