import numpy as np
class V:
    def __init__(self, d, f=()):
        self.d, self.g, self.p = np.array(d), 0.0, set(f)
        self.r = lambda: None
    def __add__(self, o):
        o = o if isinstance(o, V) else V(o)
        s = V(self.d + o.d, (self, o))
        def _r():
            self.g += s.g; o.g += s.g
        s.r = _r
        return s
    def backward(self):
        t, v = [], set()
        def b(n):
            if n not in v:
                v.add(n); [b(p) for p in n.p]; t.append(n)
        b(self); self.g = 1.0
        for n in reversed(t): n.r()
if __name__ == "__main__":
    a, b = V(2.0), V(3.0)
    c = a + b
    c.backward()
    print(a.g)
