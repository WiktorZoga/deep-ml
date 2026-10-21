import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    # Your code here
    
    params = x0.copy()

    m = np.zeros_like(params)
    v = np.zeros_like(params)

    for t in range(num_iterations):
        g = grad(params)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)

        m_hat = m / (1.0 - (beta1 ** (t + 1)))
        v_hat = v / (1.0 - (beta2 ** (t + 1)))

        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return params
