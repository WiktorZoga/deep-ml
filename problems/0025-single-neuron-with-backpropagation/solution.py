import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

	x = features
	y = labels

	w = initial_weights.copy()
	b = initial_bias

	def mse(pred, y):
		return np.mean((pred - y)**2)

	def sigmoid(z):
		return 1.0 / (1.0 + np.exp(-z))

	mse_values = []

	N = x.shape[0]

	for _ in range(epochs):
		z = x @ w + b
		pred = sigmoid(z)
		loss = mse(pred, y)

		mse_values.append(loss.round(4))

		# now dw and db

		dloss_dpred = 2.0 / N * (pred - y)
		dpred_dz = pred * (1 - pred)
		dloss_dz = dloss_dpred * dpred_dz

		dloss_dw = x.T @ dloss_dz # tranposed, but size=1
		dloss_db = np.sum(dloss_dz)

		w = w - learning_rate * dloss_dw
		b = b - learning_rate * dloss_db

		# dloss/dw = dloss/dz * dz/dw
		# dloss/dz = dloss/dpred * dpred/dz


		# dpred/dz = dsigmoid(z) = (1 - sigmoid(z)) = 1 - pred

		# sigmoid(z) = (1 + e^-z)^-1
		# dsigomid = (1 + e^-z) * e^-z = (1 - sigmoid(z))

		# dloss/dpred = 2/N * (pred - y)


	return w.round(4), b.round(4), mse_values