# Lee, D. D., & Seung, H. S. (2001). Algorithms for non-
# negative matrix factorization. In Advances in neural information
# processing systems (pp. 556-562).

import numpy as np
import nmf.mult
import nmf.pgrad
from nmf.norms import norm_Frobenius
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

B = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])

inner_dim=5
A = np.random.rand(10, 8)
W_init = np.random.randint(1, 10, (A.shape[0], inner_dim)).astype(float)
H_init = np.random.randint(1, 10, (inner_dim, A.shape[1])).astype(float)
print("W_init", W_init)
print("H_init", H_init)


W, H, errors1 = nmf.mult.factorise_Fnorm(A, inner_dim, record_errors=True,
                                         n_steps=100,
                                         W_init=W_init.copy(),
                                         H_init=H_init.copy())

W, H, errors2 = nmf.pgrad.factorise_Fnorm_subproblems_pgrad(A, inner_dim, record_errors=True,
                                          n_steps=100, epsiolon=0,
                                          W_init=W_init.copy(),
                                          H_init=H_init.copy())

W, H, errors3 = nmf.pgrad.factorise_Fnorm_direct_pgrad(A, inner_dim, record_errors=True,
                                          n_steps=1000, epsiolon=0,
                                          W_init=W_init.copy(),
                                          H_init=H_init.copy())

print(W @ H)

errors = [errors1, errors2, errors3]
labels = ["mult", "subproblems_pgrad", "direct_pgrad"]

rank_1_list = from_WH_to_rank_1_list(W, H)


def matrix_key(A):
    return tuple(np.round(A, 4).ravel())


rank_1_list = sorted(rank_1_list, key=matrix_key, reverse=True)

for i, m in enumerate(rank_1_list):
    print(i)
    print(m)

for err, lbl in zip(errors, labels):
    print(err)
    plt.plot(np.log(err), label=lbl)
plt.legend()
plt.show()
