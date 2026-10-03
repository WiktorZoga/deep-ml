def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    # P(Y = y | X = x) = P(Y = y and X = x) / P(X = x)
    x_count, xy_count = 0, 0
    for (X, Y) in data:
      if X == x:
        if Y == y:
          xy_count += 1
        x_count += 1

    x_count = max(x_count, 1)
    
    return xy_count / x_count
    