import numpy as np
def conv2d(i, f):
    ih, iw = i.shape
    fh, fw = f.shape
    sh, sw = ih - fh + 1, iw - fw + 1
    saida = np.zeros((sh, sw))
    for r in range(sh):
        for c in range(sw):
            saida[r, c] = np.sum(i[r:r+fh, c:c+fw] * f)
    return saida
if __name__ == "__main__":
    img = np.random.randn(10, 10)
    flt = np.random.randn(3, 3)
    print(conv2d(img, flt).shape)
