def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	
	return [[scalar * elem for elem in row] for row in matrix]