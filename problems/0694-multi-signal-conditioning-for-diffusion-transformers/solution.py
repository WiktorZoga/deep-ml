import numpy as np

def multi_signal_conditioning(features: np.ndarray, signals: list, proj_weights: list, proj_biases: list, mod_weight: np.ndarray, mod_bias: np.ndarray) -> np.ndarray:
	"""
	Modulate hidden features using multiple conditioning signals.

	Args:
		features: (N, D) hidden feature array
		signals: list of K conditioning signal arrays
		proj_weights: list of K projection weight matrices
		proj_biases: list of K projection bias vectors
		mod_weight: modulation weight matrix
		mod_bias: modulation bias vector

	Returns:
		(N, D) modulated features
	"""
	
	c = np.sum([s @ p + b for s, p, b in zip(signals, proj_weights, proj_biases)], axis=0)

	D = features.shape[1]

	mod_params = c @ mod_weight + mod_bias

	gamma = mod_params[:, :D]
	beta = mod_params[: , D:]

	output = (1.0 + gamma) * features + beta
	return output