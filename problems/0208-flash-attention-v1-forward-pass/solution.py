import numpy as np
import math

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                                  block_size: int = 2) -> np.ndarray:
    
    seq_len, d_model = Q.shape
    scale = 1.0 / math.sqrt(d_model)
    n_blocks = seq_len // block_size

    Q_blocks = Q.reshape(n_blocks, block_size, d_model)
    K_blocks = K.reshape(n_blocks, block_size, d_model)
    V_blocks = V.reshape(n_blocks, block_size, d_model)

    O_blocks = np.zeros((n_blocks, block_size, d_model))
    L_blocks = np.zeros((n_blocks, block_size, 1))
    M_blocks = np.full((n_blocks, block_size, 1), -np.inf)

    for j in range(n_blocks):	
        K_j = K_blocks[j] 
        V_j = V_blocks[j] 
        
        for i in range(n_blocks):	
            Q_i = Q_blocks[i] 
            
            O_i = O_blocks[i] 
            M_i = M_blocks[i] 
            L_i = L_blocks[i] 

            S_ij = scale * (Q_i @ K_j.T) 
            M_ij = np.max(S_ij, axis=-1, keepdims=True) 

            P_ij_unnormalized = np.exp(S_ij - M_ij) 
            L_ij = np.sum(P_ij_unnormalized, axis=-1, keepdims=True) 

            M_i_new = np.maximum(M_i, M_ij)

            a = np.exp(M_i - M_i_new)
            b = np.exp(M_ij - M_i_new)

            L_i_new = (L_i * a) + (L_ij * b)

            O_i_new = (O_i * a) + (P_ij_unnormalized @ V_j) * b

            O_blocks[i] = O_i_new
            M_blocks[i] = M_i_new
            L_blocks[i] = L_i_new

    O_blocks = O_blocks / L_blocks

    return O_blocks.reshape(seq_len, d_model)
