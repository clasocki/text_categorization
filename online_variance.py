def online_variance(data):
    n = 0
    mean = 0.0
    M2 = 0.0
     
    for x in data:
        n += 1
        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta*delta2

    if n < 2:
        return float('nan')
    else:
        return M2 / (n - 1)
