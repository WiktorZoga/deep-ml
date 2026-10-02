import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
	"""
	Shift and scale values from their original range [min, max] to a target [c, d] range.
	"""
	if values.size == 0:
		return values

	a = np.min(values)
	b = np.max(values)

	if a == b:
		return np.full_like(values, c)

	t = (values - a) / (b - a)
	
	return t * (d - c) + c
