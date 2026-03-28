import numpy as np
def mp(x, w, t):
    return 1 if np.sum(x * w) >= t else 0
if __name__ == "__main__":
    w = np.array([1, 1])
    t = 1.5
    print(f"AND(1,1) = {mp(np.array([1,1]), w, t)}")
    print(f"AND(1,0) = {mp(np.array([1,0]), w, t)}")
