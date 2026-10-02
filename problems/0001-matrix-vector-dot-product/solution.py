def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	res = []

	for row in a:
		dot = 0
		
		if len(row) != len(b):
			return -1;

		for (x, y) in zip(row, b):
			dot += x * y
		res.append(dot)
	
	return res
