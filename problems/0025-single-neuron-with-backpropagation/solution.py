import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(x, w, b):
        z = x @ w + b
        return sigmoid(z), z

    def loss(pred, y):
        return float(np.mean((pred - y) ** 2))

    def backward(x, w, b, z, y_pred, y):
        n = x.shape[0]

        dl_dy_pred = (2.0 / n) * (y_pred - y)

        dy_pred_dz = y_pred * (1.0 - y_pred)

        dl_dz = dl_dy_pred * dy_pred_dz

        dl_dw = x.T @ dl_dz
        dl_db = float(np.sum(dl_dz))

        return dl_dw, dl_db

    w = initial_weights.copy()
    b = float(initial_bias)
    mse_values = []

    y = labels.reshape(-1) if w.ndim == 1 else labels

    for _ in range(epochs):
        y_pred, z = forward(features, w, b)

        current_loss = loss(y_pred, y)
        mse_values.append(current_loss)

        dl_dw, dl_db = backward(features, w, b, z, y_pred, y)

        w -= learning_rate * dl_dw
        b -= learning_rate * dl_db

    return w, b, mse_values