class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """

        self.vocab = ['<BOS>', '<EOS>'] + list(sorted(set((list(text)))))
        self.vocab_size = len(self.vocab)

        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}


    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """

        return [0] + [self.stoi[t] for t in list(text)] + [1]
        
    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        return ''.join([self.itos[i] for i in indices])

# bos eos e h l o
# 0   1   2 3 4 5

# hello
# 0 3 2 4 4 5 1