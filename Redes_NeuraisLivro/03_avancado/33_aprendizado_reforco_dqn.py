import numpy as np
class DQN:
    def __init__(self, s, a):
        self.w = np.random.randn(s, a)
    def pred(self, x): return np.dot(x, self.w)
if __name__ == "__main__":
    dqn = DQN(4, 2)
    print(dqn.pred(np.random.randn(1, 4)))
