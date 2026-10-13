def replace_pair(sequences, pair, new_id):
    # Your code here

    new_sequences = []

    for seq in sequences:
        new_seq = []
        i = 0
        n = len(seq)
        while i < n:
           
            if i + 1 < n and (seq[i], seq[i + 1]) == pair:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_sequences.append(new_seq)
    
    return new_sequences
