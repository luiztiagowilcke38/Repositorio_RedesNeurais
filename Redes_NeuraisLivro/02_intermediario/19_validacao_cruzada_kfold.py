import numpy as np
def kfold(data, k=5):
    np.random.shuffle(data)
    return np.array_split(data, k)
if __name__ == "__main__":
    d = np.arange(20).reshape(10, 2)
    folds = kfold(d)
    for i, f in enumerate(folds): print(f"Fold {i}: {len(f)}")
