import numpy as np
def pooling(i, f=2, s=2, m="max"):
    h, w = i.shape
    oh, ow = (h - f)//s + 1, (w - f)//s + 1
    sa = np.zeros((oh, ow))
    for r in range(oh):
        for c in range(ow):
            j = i[r*s:r*s+f, c*s:c*s+f]
            sa[r,c] = np.max(j) if m=="max" else np.mean(j)
    return sa
if __name__ == "__main__":
    img = np.random.randn(4, 4)
    print(pooling(img))
