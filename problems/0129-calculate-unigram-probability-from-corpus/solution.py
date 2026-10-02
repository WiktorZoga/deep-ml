def unigram_probability(corpus: str, word: str) -> float:
    # Your code here
    tokens = corpus.split()

    if len(tokens) == 0:
        return 0.0

    count = tokens.count(word)

    return count / len(tokens)