import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        T, D = x.shape
        
        h = initial_hidden_state.copy()
        c = initial_cell_state.copy()

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        hidden_states = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)  
            
            z = np.concatenate([h, x_t], axis=0)

            f = sigmoid(self.Wf @ z + self.bf) 
            i = sigmoid(self.Wi @ z + self.bi) 
            c_hat = np.tanh(self.Wc @ z + self.bc)

            c = f * c + i * c_hat

            o = sigmoid(self.Wo @ z + self.bo)

            h = o * np.tanh(c)
            
            hidden_states.append(h)
        
        outputs = np.array(hidden_states)
        
        return outputs, h, c