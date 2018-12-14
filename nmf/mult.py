import nmf.norms
import numpy as np


def factorise_Fnorm(V, n, n_steps=10000, min_err=1e-6, record_errors=False):
    W = 1 - np.random.rand(V.shape[0], n)
    H = 1 - np.random.rand(n, V.shape[1])
    errors = []

    for i in range(n_steps):
        err = nmf.norms.norm_Frobenius(V, W @ H)
        if record_errors:
            errors.append(err)

        if err < min_err:
            if record_errors:
                return W, H, errors
            else:
                return W, H

        W = update_first_Fnorm(V, W, H)
        H = update_second_Fnorm(V, W, H)

    if record_errors:
        return W, H, errors
    else:
        return W, H

def update_first_Fnorm(V, W, H):
    VHt = V  @ H.T
    # TODO chack order for better performance
    WHHt = W @ H @ H.T

    W = W * VHt / WHHt

    return W


def update_second_Fnorm(V, W, H):
    WtV = W.T @ V

    # TODO chack order for better performance
    WtWH = W.T @ W @ H

    H = H * WtV / WtWH

    return H


def factorise_KLdiv(A, n):
    pass


def update_first_KDdiv(V, W, H):
    WH = V  @ H.T
    H_sums = np.sum(H, axis=1)

    # TODO chack order for better performance
    WHHt = W @ H @ H.T

    W = W * VHt / WHHt

    return W