import numpy as np
import nmf.mult
import nmf.pgrad
import nmf.nesterov
import nmf.dtpnn
import nmf.bayes

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

np.random.seed(3)

inner_dim = 4
size = (6, 5)
A_unnoised = np.random.rand(size[0], inner_dim) @ np.random.rand(inner_dim, size[1])
A = A_unnoised + 1/2 * np.random.rand(*size)
# A_unnoised = A

W_init = np.random.rand(*(A.shape[0], inner_dim)).astype(float)
H_init = np.random.rand(*(inner_dim, A.shape[1])).astype(float)

print("W_init", W_init)
print("H_init", H_init)

def f():
    W, H, errors2 = nmf.pgrad.factorise_Fnorm_subproblems(A, inner_dim, record_errors=True,
                                              n_steps=100, epsiolon=0,
                                              W_init=W_init.copy(),
                                              H_init=H_init.copy())

    W, H, errors3 = nmf.nesterov.factorise_Fnorm(A, inner_dim, record_errors=True,
                                              n_steps=100, epsiolon=0,
                                              W_init=W_init.copy(),
                                              H_init=H_init.copy())

W, H, errors1 = nmf.mult.factorise_Fnorm(A, inner_dim, record_errors=True,
                                         n_steps=10000,
                                         W_init=W_init.copy(),
                                         H_init=H_init.copy(),
                                         min_err=0)

W, H, errors4 = nmf.bayes.factorise_ICM(A, inner_dim, record_errors=True,
                                             n_steps=1000, min_err=0,
                                            W_prior=W_init.copy(),
                                            H_prior=H_init.copy())

samples_W, samples_H, errors5 = nmf.bayes.factorise_Gibbs(A, inner_dim, record_errors=True,
                                          n_steps=8000,
                                          W_prior=np.ones(W_init.shape),
                                          H_prior=np.ones(H_init.shape))

W_mean = np.mean(samples_W, axis=0)
H_mean = np.mean(samples_H, axis=0)

plt.figure()
plt.subplot(141)
plt.imshow(A_unnoised); plt.title("A_unnoised");

plt.subplot(142)
plt.imshow(A); plt.title("A");
plt.subplot(143)
plt.imshow(W_mean @ H_mean); plt.title("W_mean @ H_mean");

W = samples_W[-1, :, :]
H = samples_H[-1, :, :]
plt.subplot(144)
plt.imshow(W @ H); plt.title("W @ H last");

W = samples_W[-1, :, :]
H = samples_H[-1, :, :]
plt.subplot(144)
plt.imshow(W @ H); plt.title("CI size");


def get_ci_size(samples, ci_size=80):
    offset = (100 - ci_size) / 2
    A_min = np.percentile(samples, offset, axis=0, interpolation="nearest")
    A_max = np.percentile(samples, 100-offset, axis=0, interpolation="nearest")
    return A_max - A_min

plt.figure()
plt.subplot(221)
plt.imshow(W_mean); plt.title("W_mean");
plt.subplot(223)
plt.imshow(H_mean); plt.title("H_mean");
plt.subplot(222)
plt.imshow(get_ci_size(samples_W)); plt.title("W_get_ci_size");
plt.subplot(224)
plt.imshow(get_ci_size(samples_H)); plt.title("H_get_ci_size");


print(W @ H)

errors = [
    errors1,
    # errors2, errors3,
    errors4,
    errors5]
labels = [
    "mult",
    # "subproblems_pgrad", "nesterov",
    "bayes ICM",
    "bayes GIBBS"]

rank_1_list = from_WH_to_rank_1_list(W, H)


def matrix_key(A):
    return tuple(np.round(A, 4).ravel())


rank_1_list = sorted(rank_1_list, key=matrix_key, reverse=True)

for i, m in enumerate(rank_1_list):
    print(i)
    print(m)

plt.figure()
for err, lbl in zip(errors, labels):
    print(err)
    plt.plot(err[:, 1], np.log(err[:,0] / (A.shape[0] * A.shape[1])), label=lbl)
plt.legend()
plt.show()


# prior
l = np.array([[0.1, 0.2],
              [0.3, 0.4]])

# posterior
s = np.array([[0.1, 0.2],
              [0.3, 0.4]])

m = np.array([[10., 5.],
              [3., 2.]])


