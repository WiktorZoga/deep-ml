import math

def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.

    Args:
        obs_space: dict with 'type' and type-specific keys defining the observation space
        act_space: dict with 'type' and type-specific keys defining the action space
        hidden_layers: list of ints specifying hidden layer widths

    Returns:
        dict with keys:
          'input_dim': int
          'output_dim': int
          'layer_shapes': list of ((weight_rows, weight_cols), (bias_dim,)) tuples
          'total_params': int
    """
    if obs_space['type'] == 'Box':
        input_dim = math.prod(obs_space['shape'])
    elif obs_space['type'] == 'Discrete':
        input_dim = obs_space['n']
    elif obs_space['type'] == 'MultiBinary':
        input_dim = obs_space['n']
    
    if act_space['type'] == 'Box':
        output_dim = math.prod(act_space['shape'])
    elif act_space['type'] == 'Discrete':
        output_dim = act_space['n']
    elif act_space['type'] == 'MultiBinary':
        output_dim = act_space['n']

    layers = [input_dim] + hidden_layers + [output_dim]

    layer_shapes = [((in_dim, out_dim), (out_dim,)) for (in_dim, out_dim) in zip(layers[:-1], layers[1:])]

    total_params = sum([layer_shape[1][0] * (1 + layer_shape[0][0]) for layer_shape in layer_shapes])

    return {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'layer_shapes': layer_shapes,
        'total_params': total_params,
    }


        
