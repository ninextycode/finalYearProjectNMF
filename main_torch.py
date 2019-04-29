# rough work & testing

import numpy as np

import torch
import matplotlib.pyplot as plt


np.set_printoptions(precision=3, suppress=True)


A = torch.tensor([
    [0.35, 1, 1, 1, 1],
    [1,    1, 1, 0, 0],
    [0,    0, 1, 1, 0],
    [0,    0, 0, 1, 1],
    [0,    1, 0, 0, 1]
])

inner_dim = 4

W, H, errors = nmf_torch.nesterov.factorise_Fnorm(A, inner_dim,
                                                  record_errors=True,
                                                  n_steps=100,
                                                  epsilon=0)

plt.figure()
plt.subplot(121)
plt.imshow(A)
plt.title("A")
plt.subplot(122)
plt.imshow(W @ H)
plt.title("W @ H")

plt.figure()
plt.plot(errors[:, 1], np.log(errors[:,0] / (A.shape[0] * A.shape[1])))

plt.show()

