import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:

    x = np.asarray(input_sequence)
    h = np.asarray(initial_hidden_state)

    t, _ = x.shape

    Wx = np.asarray(Wx)
    Wh = np.asarray(Wh)
    b = np.asarray(b)

    for i in range(t):
        z = (Wx @ x[i]) + (Wh @ h) + b
        h = np.tanh(z)
	
    return h.tolist()
