import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""

	val = (1.0 / (math.sqrt(2.0 * math.pi) * std_dev)) * math.exp(-(mean - x)**2 / (2.0 * std_dev**2))

	return round(val,5)