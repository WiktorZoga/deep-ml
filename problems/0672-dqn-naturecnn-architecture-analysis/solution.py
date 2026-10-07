def nature_cnn_info(input_shape: tuple, num_actions: int) -> dict:
    """
    Compute architecture details for the NatureCNN used in deep Q-networks.

    Args:
        input_shape: tuple of (channels, height, width)
        num_actions: number of output actions

    Returns:
        dict with architecture details
    """

    def conv_output_shape(in_shape, layer):
        channels = layer["filters"]
        kernel = layer["kernel"]
        stride = layer["stride"]
        padding = layer.get("padding", 0)

        h, w = in_shape[1], in_shape[2]
        h_out = (h - kernel + 2 * padding) // stride + 1
        w_out = (w - kernel + 2 * padding) // stride + 1

        return (channels, h_out, w_out)

    conv_layers = [
        {"filters": 32, "kernel": 8, "stride": 4},
        {"filters": 64, "kernel": 4, "stride": 2},
        {"filters": 64, "kernel": 3, "stride": 1},
    ]

    stats = {}
    current_shape = input_shape

    for i, layer in enumerate(conv_layers, start=1):
        current_shape = conv_output_shape(current_shape, layer)
        stats[f"conv{i}_output_shape"] = current_shape

    last_conv_shape = stats["conv3_output_shape"]
    stats["flatten_size"] = last_conv_shape[0] * last_conv_shape[1] * last_conv_shape[2]
    stats["fc_output_size"] = 512
    stats["output_size"] = num_actions

    in_channels = input_shape[0]
    total_params = 0

    for layer in conv_layers:
        total_params += layer["filters"] * (in_channels * (layer["kernel"] ** 2) + 1)
        in_channels = layer["filters"]

    # Parametry warstwy FC (512 neuronów z wejścia flatten)
    total_params += stats["fc_output_size"] * (stats["flatten_size"] + 1)

    # Parametry warstwy wyjściowej (num_actions neuronów z wejścia 512)
    total_params += stats["output_size"] * (stats["fc_output_size"] + 1)

    stats["total_params"] = total_params

    return stats