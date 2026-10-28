import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    
    parameter = x0
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)

    for t in range(1, num_iterations+1):
        g = grad(parameter)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g**2

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        parameter = parameter - learning_rate * m_hat / np.sqrt(v_hat + epsilon)
    
    return parameter


