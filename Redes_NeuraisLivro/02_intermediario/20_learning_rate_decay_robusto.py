import numpy as np
lr, decay = 0.1, 0.01
for e in range(10):
    lr_at = lr * (1 / (1 + decay * e))
    print(f"Epoca {e}, LR: {lr_at}")
