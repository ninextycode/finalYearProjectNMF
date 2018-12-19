# Lee, D. D., & Seung, H. S. (2001). Algorithms for non-
# negative matrix factorization. In Advances in neural information
# processing systems (pp. 556-562).

import numpy as np
from nmf.mult import factorise_Fnorm, factorise_KLdiv
from nmf.represent import from_WH_to_rank_1_list

import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)


A = np.array([
    [0.35, 1, 1, 1, 1],
    [1,    1, 1, 0, 0],
    [0,    0, 1, 1, 0],
    [0,    0, 0, 1, 1],
    [0,    1, 0, 0, 1]
])

W, H, errors1 = factorise_KLdiv(A, 4, record_errors=True, n_steps=300)
W, H, errors2 = factorise_Fnorm(A, 4, record_errors=True, n_steps=300)



rank_1_list = from_WH_to_rank_1_list(W, H)


def matrix_key(A):
    return tuple(np.round(A, 4).ravel())


rank_1_list = sorted(rank_1_list, key=matrix_key, reverse=True)

for i, m in enumerate(rank_1_list):
    print(i)
    print(m)

plt.plot(np.log(errors1))
plt.plot(np.log(errors2))
plt.show()
