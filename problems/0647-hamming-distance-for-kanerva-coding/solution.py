import numpy as np

def hamming_distance_kanerva(state: list, prototypes: list, threshold: int) -> tuple:
	"""
	Compute Hamming distances and find active prototypes for Kanerva coding.

	Args:
		state: Binary state vector (list of 0s and 1s).
		prototypes: List of binary prototype vectors.
		threshold: Maximum Hamming distance for a prototype to be active.

	Returns:
		Tuple of (distances, active_indices).
	"""

	n = len(prototypes)

	state = np.asarray(state)
	prototypes = np.asarray(prototypes)

	hamming = np.sum(np.abs(prototypes - state), axis=1)

	active = hamming <= threshold

	return hamming.tolist(), np.arange(n)[active].tolist()