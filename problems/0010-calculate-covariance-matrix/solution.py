def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here

	def cov(X: list[float], Y: list[float]) -> float:
		n = len(X)

		mean_x = sum(X) / n
		mean_y = sum(Y) / n

		return sum([(x - mean_x) * (y - mean_y) for (x, y) in zip(X, Y)]) / (n - 1)
	
	n = len(vectors)

	res = []

	for i in range(n):
		cnt = []
		for j in range(n):
			cnt.append(cov(vectors[i], vectors[j]))
		res.append(cnt)

	return  res



