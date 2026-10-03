import numpy as np

def analyze_memory_fragmentation(block_status: list) -> dict:
	"""
	Analyze memory fragmentation in a block-based memory pool.

	Args:
		block_status: List of ints where 1 = allocated block, 0 = free block

	Returns:
		dict with keys:
			'utilization': float, fraction of allocated blocks
			'num_free_fragments': int, count of contiguous free regions
			'largest_free_fragment': int, size of largest contiguous free region
			'fragmentation_ratio': float, measure of free memory scatter
	"""
	
	status = np.asarray(block_status)

	utilization = np.mean(status)

	free_blocks = len(block_status) - np.sum(status)

	pre = -1
	cnt_len = 0
	max_len = 0
	num_free_fragments = 0
	for s in block_status:
		if s == 0:
			cnt_len += 1
			if pre != 0:
				num_free_fragments += 1
		else:
			cnt_len = 0
		max_len = max(max_len, cnt_len)
		pre = s
	
	largest_free_fragment = max_len

	fragmentation_ratio = 1.0 - largest_free_fragment / free_blocks if free_blocks > 0 else 0.0

	return {
		"utilization": utilization.round(4),
		"num_free_fragments": num_free_fragments,
		"largest_free_fragment": largest_free_fragment,
		"fragmentation_ratio": round(fragmentation_ratio, 4),
	}
