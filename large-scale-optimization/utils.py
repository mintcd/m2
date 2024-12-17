import numpy as np

def gradient(f, x, dist=1e-3):
    if type(x) is np.ndarray:
        dim = x.shape[0]
        grad_f = np.zeros(dim)

        for i in range(dim):
            diff = np.zeros(dim)
            diff[i] = dist

            grad_f[i] = (f(x + diff) - f(x - diff)) / (2 * dist)

        return grad_f
    else:
        return (f(x + dist) - f(x - dist)) / (2 * dist)