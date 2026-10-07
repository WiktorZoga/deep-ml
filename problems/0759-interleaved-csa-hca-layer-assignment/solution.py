def assign_layer_types(num_layers: int, num_dense_hca: int) -> list:
    """Return a list of 'HCA'/'CSA' strings per layer."""
    
    def choose_type(i):
        if i < num_dense_hca or i % 2 == 1:
            return 'HCA'
        return 'CSA'
    
    return [choose_type(i) for i in range(num_layers)]