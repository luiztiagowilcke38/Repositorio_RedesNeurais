import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
w, b, lr = 0.0, 0.0, 0.01
for _ in range(100):
    pred = w * x + b
    dw = -2 * np.dot(x, (y - pred)) / len(x)
    db = -2 * np.sum(y - pred) / len(x)
    w -= lr * dw
    b -= lr * db
print(f"w: {w}, b: {b}")
