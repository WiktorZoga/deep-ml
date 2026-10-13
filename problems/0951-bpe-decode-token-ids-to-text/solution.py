def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    tokens = []
    
    for i in ids:
        value = vocab[i]
        
        if value.startswith('G'):
            value = ' ' + value[1:]
            
        tokens.append(value)

    return ''.join(tokens)