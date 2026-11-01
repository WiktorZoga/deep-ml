import numpy as np

def conv3d_forward_pass(
    input_volume: np.ndarray,
    kernel: np.ndarray,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
	"""
	Perform 3D convolution forward pass.
	
	Slide a 3D kernel over input volume, computing dot products.
	
	Args:
		input_volume: Shape (C, D, H, W)
		  C = channels, D = depth/time, H = height, W = width
		kernel: Shape (C, kD, kH, kW)
		  Must match input channels
		stride: (stride_d, stride_h, stride_w)
		  Step size in each dimension
		padding: (pad_d, pad_h, pad_w)
		  Zero-padding in each dimension
	
	Returns:
		Output volume: Shape (1, D_out, H_out, W_out)
		  Single output channel
		
	Process:
		1. Apply padding to input
		2. Calculate output dimensions
		3. For each output position:
		   - Extract 3D patch from input
		   - Compute element-wise product with kernel
		   - Sum all products -> single output value
	"""

	C, D, H, W = input_volume.shape
	_, kD, kH, kW = kernel.shape

	D_out = (D - kD + 2 * padding[0]) // stride[0] + 1
	H_out = (H - kH + 2 * padding[1]) // stride[1] + 1
	W_out = (W - kW + 2 * padding[2]) // stride[2] + 1

	pad_width = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]))

	x = np.pad(input_volume, pad_width=pad_width, mode='constant', constant_values=0)

	out = np.zeros((1, D_out, H_out, W_out))

	for i in range(D_out):
		depth_start = i * stride[0]
		depth_end = depth_start + kD
		for j in range(H_out):
			row_start = j * stride[1]
			row_end = row_start + kH
			for k in range(W_out):
				col_start = k * stride[2]
				col_end = col_start + kW

				window = x[:, depth_start:depth_end, row_start:row_end, col_start:col_end]

				out[0, i, j, k] = np.sum(window * kernel)
	return out


