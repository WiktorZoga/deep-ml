def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:

    n = len(a)
    m = len(b[0])
    d_a = len(a[0])
    d_b = len(b)

    if d_a != d_b:
        return -1

    d = d_a

    c = []

    for i in range(n):
        c.append([0.0] * m)
        for j in range(m):
            for k in range(d):
                c[i][j] += a[i][k] * b[k][j]
    return c

