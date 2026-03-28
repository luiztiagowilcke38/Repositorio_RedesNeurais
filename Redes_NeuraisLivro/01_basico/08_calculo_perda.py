import numpy as np
def mse(y, p): return np.mean((y - p)**2)
def mae(y, p): return np.mean(np.abs(y - p))
if __name__ == "__main__":
    y, p = np.array([1, 0, 1]), np.array([0.9, 0.1, 0.8])
    print(f"MSE: {mse(y, p)}, MAE: {mae(y, p)}")
