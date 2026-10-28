import numpy as np

def pos_encoding(position: int, d_model: int):

	if position == 0 or d_model <= 0:
		return -1

	i = np.arange(d_model // 2)

	positions = np.arange(position).reshape(position, 1)

	w = np.exp(-np.log(10000.) * (2.0 * i / d_model))

	t = positions * w

	pos_encoding = np.empty((position, d_model))

	pos_encoding[:, 0::2] = np.sin(t)
	pos_encoding[:, 1::2] = np.cos(t)

	pos_encoding = np.float16(pos_encoding)
	return pos_encoding