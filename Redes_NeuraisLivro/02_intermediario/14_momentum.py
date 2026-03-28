import numpy as np
w, lr, mom, v = 10.0, 0.1, 0.9, 0.0
for _ in range(100):
    grad = 2 * w
    v = mom * v - lr * grad
    w += v
print(w)
