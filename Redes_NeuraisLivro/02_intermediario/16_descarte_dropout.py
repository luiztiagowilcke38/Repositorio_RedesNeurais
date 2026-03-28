import numpy as np
def dropout(x, p=0.5):
    mask = np.random.binomial(1, 1-p, size=x.shape) / (1-p)
    return x * mask
if __name__ == "__main__":
    x = np.ones((1, 10))
    print(dropout(x))
