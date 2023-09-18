import numpy as np

a = [0 for _ in range(28)]
l = [1, 1, 1] + [0 for _ in range(25)]
r = [0 for _ in range(25)] + [1, 1, 1]

x = np.array([l, l, l] + [a for _ in range(22)] + [r, r, r])
print(x.shape)