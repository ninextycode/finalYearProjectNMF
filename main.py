import numpy as np
from nmf.mult import factorise_Fnorm
from nmf.norms import norm_Frobenius
from nmf.represent import from_WH_to_rank_1_list

import matplotlib.pyplot as plt

A = np.array([
    [1.35, 1, 1, 1, 1],
    [1,    1, 1, 0, 0],
    [0,    0, 1, 1, 0],
    [0,    0, 0, 1, 1],
    [0,    1, 0, 0, 1]
])

W, H, errors = factorise_Fnorm(A, 4, record_errors=True)

np.set_printoptions(precision=3, suppress=True)

print(np.round(W, 4))
print(np.round(H, 4))
print(np.round(W @ H, 4))
print(norm_Frobenius(A, W @ H))

rank_1_list = from_WH_to_rank_1_list(W, H)


def matrix_key(A):
    return tuple(np.round(A, 4).ravel())


rank_1_list = sorted(rank_1_list, key=matrix_key, reverse=True)

for i, m in enumerate(rank_1_list):
    print(i, m)

plt.plot(errors)
plt.show()