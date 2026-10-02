import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
	"""
	Update parameters using the Adamax optimizer.
	"""
	m_next = beta1 * m + (1 - beta1) * grad
	
	u_next = np.maximum(beta2 * u, np.abs(grad))
	
	m_hat = m_next / (1.0 - beta1 ** t)
	
	parameter_next = parameter - (learning_rate / (u_next + epsilon)) * m_hat

	return np.round(parameter_next, 5), np.round(m_next, 5), np.round(u_next, 5)
