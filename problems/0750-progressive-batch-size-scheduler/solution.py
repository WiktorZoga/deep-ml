import numpy as np

def progressive_batch_size(
    queries: list[int], milestones: list[int], batch_sizes: list[int]
) -> list[int]:
    """
	Return the active batch size at each queried token count.

	Args:
		queries: List of token counts (tokens processed so far) to query.
		milestones: Ascending token-count thresholds where the batch size steps up.
		batch_sizes: Batch sizes for each stage. Length must be len(milestones) + 1.

	Returns:
		List of batch sizes corresponding to each query.
	"""
    queries_arr = np.asarray(queries, dtype=int)
    milestones_arr = np.asarray(milestones, dtype=int)
    batch_sizes_arr = np.asarray(batch_sizes, dtype=int)

    indices = np.searchsorted(milestones_arr, queries_arr, side="right")

    return batch_sizes_arr[indices].tolist()