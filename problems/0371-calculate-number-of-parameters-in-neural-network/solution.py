def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	# Your code here
	def calculate(layer: dict) -> int:
		if layer["type"] == "dense":
			f_in = layer["input_size"]
			f_out = layer["output_size"]
			bias = layer["bias"] if "bias" in layer else True

			params = (f_in + (1 if bias else 0)) * f_out

		elif layer["type"] == "conv2d":
			ch_in = layer["in_channels"]
			ch_out = layer["out_channels"]
			kernel_size = layer["kernel_size"]
			bias = layer["bias"] if "bias" in layer else True

			params = (kernel_size ** 2 * ch_in + (1 if bias else 0)) * ch_out 
		else:
			raise Exception("Not known layer type")

		return params
	
	params = sum([calculate(layer) for layer in layers])
	return params