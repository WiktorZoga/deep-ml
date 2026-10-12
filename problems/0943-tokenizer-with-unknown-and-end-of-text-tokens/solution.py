import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        pattern = r"([,.:;?_!\"()']|--|\s)"
        
        pieces = re.split(pattern, text)

        tokens = [piece.strip() for piece in pieces if piece and piece.strip()]

        encoded_ids = [vocab.get(token, vocab['<|unk|>']) for token in tokens]
        
        return encoded_ids


    else:
        inv_vocab = {v: k for k, v in vocab.items()}

        tokens = [inv_vocab[idx] for idx in text]

        code = ' '.join(tokens)

        code = re.sub(r"\s+([,.?!\"()'])", r"\1", code)

        return code
