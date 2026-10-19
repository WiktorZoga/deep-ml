import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):

	x_padded = np.pad(input_matrix, pad_width=padding, mode='constant', constant_values=0)

	padded_height, padded_width = x_padded.shape
	kernel_height, kernel_width = kernel.shape

	out_height = (padded_height - kernel_height) // stride + 1
	out_width = (padded_width - kernel_width) // stride + 1

	output = np.zeros(((out_height, out_width)))

	for i in range(out_height):
		for j in range(out_width):
			row_start = i * stride
			row_end = row_start + kernel_height

			col_start = j * stride
			col_end = col_start + kernel_width

			windows = x_padded[row_start:row_end, col_start:col_end]

			output[i, j] = np.sum(windows * kernel)

	return output