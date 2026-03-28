import numpy as np
w, lr = 10.0, 0.1
for _ in range(100):
    grad = 2 * w
    w -= lr * grad
print(w)
