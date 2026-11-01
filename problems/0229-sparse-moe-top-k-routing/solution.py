import numpy as np

def moe_topk_routing(
    router_logits: np.ndarray,
    expert_outputs: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    
    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs
    
    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token
        
    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """

    batch_size = router_logits.shape[0]

    top_experts = np.argsort(-router_logits, axis=-1, kind='stable')[:, :k]

    batch_indices = np.arange(batch_size)[:, None]

    top_scores = router_logits[batch_indices, top_experts]

    def softmax(z):
        z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return z / np.sum(z, axis=-1, keepdims=True)
    
    soft_scores = softmax(top_scores)

    selected_expert_outputs = expert_outputs[batch_indices, top_experts]

    routed_outputs = soft_scores[:, :, None] * selected_expert_outputs

    return np.sum(routed_outputs, axis=1)

