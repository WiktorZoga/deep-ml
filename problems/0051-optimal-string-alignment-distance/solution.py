import numpy as np

def OSA(source: str, target: str) -> int:
    n = len(source)
    m = len(target)

    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
	
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if source[i - 1] == target[j - 1] else 1

            dp[i, j] = min(
                dp[i - 1, j] + 1,       # Usuwanie
                dp[i, j - 1] + 1,       # Wstawianie
                dp[i - 1, j - 1] + cost # podmiana
            )

            if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                dp[i, j] = min(dp[i, j], dp[i - 2, j - 2] + 1)
	
    return int(dp[n, m])
