import numpy as np
w, lr, lamb = 10.0, 0.1, 0.01
for _ in range(100):
    grad = 2 * w + lamb * w
    w -= lr * grad
print(w)
