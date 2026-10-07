def collect_rollouts(seeds: list[int], grid_size: int) -> list[int]:
	"""
	Collect optimal episode returns across procedurally generated gridworlds.

	Args:
		seeds: list of integer seeds, each defining one procedural environment.
		grid_size: side length of the square grid.

	Returns:
		A list of episode returns (one per seed).
	"""
	GX = [seed % grid_size for seed in seeds]
	GY = [(seed // grid_size) % grid_size for seed in seeds]
	return [10 - (gx + gy) if (gx != 0 or gy != 0) else 10 - (grid_size * 2 - 2) for gx, gy in zip(GX, GY)]
