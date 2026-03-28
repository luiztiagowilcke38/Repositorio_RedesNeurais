import numpy as np
def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1])
    print(softmax(logits))
