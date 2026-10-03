class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """

        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        chars = list(sorted(set(text)))

        self.stoi = {token: idx + 2 for idx, token in enumerate(chars)}

        self.stoi[self.bos_token] = 0
        self.stoi[self.eos_token] = 1

        self.itos = {idx: token for token, idx in self.stoi.items()}

        self.vocab_size = len(chars) + 2


    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """

        tokens = [self.bos_token] + list(text) + [self.eos_token]

        return [self.stoi[token] for token in tokens]

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        return ''.join([self.itos[idx] for idx in indices])
