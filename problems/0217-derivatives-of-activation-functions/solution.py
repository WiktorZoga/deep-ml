import math

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here

	# sigmoid = (1 + e^{-x})^{-1}
	# d/dx sigomid = sigmoid * (1 - sigmoid)

	# d/dx tahn = 1 - tanh^2

	def sigmoid(x):
		return 1.0 / (1.0 + math.exp(-x))

	return {"sigmoid" : sigmoid(x) * (1 - sigmoid(x)),
			"tanh": 1.0 - math.tanh(x)**2,
			"relu": 1.0 if x > 0 else 0.0
			}
