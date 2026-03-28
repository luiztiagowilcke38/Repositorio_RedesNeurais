import numpy as np
class GAN:
    def __init__(self, z, x):
        self.w_g = np.random.randn(z, x)
        self.w_d = np.random.randn(x, 1)
    def g(self, z): return np.tanh(np.dot(z, self.w_g))
    def d(self, x): return 1/(1+np.exp(-np.dot(x, self.w_d)))
if __name__ == "__main__":
    gan = GAN(10, 20)
    print(gan.d(gan.g(np.random.randn(1, 10))))
