import numpy as np

def calculate_brightness(img):

	n = len(img)

	if n == 0:
		return -1
	
	m = len(img[0])

	for row in img:
		if len(row) != m:
			return -1
		mini = min(row)
		maxi = max(row)
		if mini < 0:
			return -1
		if maxi > 255:
			return -1

	# Write your code here
	img = np.asarray(img)

	return np.mean(img)