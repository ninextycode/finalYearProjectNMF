# TODO chack order for better performance

import nmf.norms
import numpy as np


def factorise_Fnorm(V, inner_dim, n_steps=10000, min_err=1e-6, record_errors=False):
    return factorise(V, inner_dim, n_steps, min_err, record_errors,
                     update_first=update_first_Fnorm,
                     update_second=update_second_Fnorm,
                     error=nmf.norms.norm_Frobenius)


def factorise_KLdiv(V, inner_dim, n_steps=10000, min_err=1e-6, record_errors=False):
    return factorise(V, inner_dim, n_steps, min_err, record_errors,
                     update_first=update_first_KLdiv,
                     update_second=update_second_KLdiv,
                     error=nmf.norms.divergence_KullbackLeible)

def update_first_Fnorm(V, W, H):
    VHt = V  @ H.T
    WHHt = W @ H @ H.T

    W = W * VHt / WHHt

    return W


def update_second_Fnorm(V, W, H):
    WtV = W.T @ V

    # TODO chack order for better performance
    WtWH = W.T @ W @ H

    H = H * WtV / WtWH

    return H


def update_first_KLdiv(V, W, H):
    WH = W @ H
    WH[WH == 0] = 1e-6
    num = (V / WH) @ H.T
    S_H = np.sum(H, keepdims=True, axis=1).T
    W = W * num / S_H
    return W


def update_second_KLdiv(V, W, H):
    WH = W @ H
    WH[WH==0] = 1e-6
    num = W.T @ (V / WH)
    S_W = np.sum(W, keepdims=True, axis=0).T
    H = H * num / S_W
    return H


def factorise(V, inner_dim, n_steps, min_err, record_errors,
              update_first, update_second, error):
    W = 1 - np.random.rand(V.shape[0], inner_dim)
    H = 1 - np.random.rand(inner_dim, V.shape[1])
    errors = []

    for i in range(n_steps):
        err = error(V, W @ H)
        if record_errors:
            errors.append(err)

        if err < min_err:
            if record_errors:
                return W, H, errors
            else:
                return W, H

        W = update_first(V, W, H)
        H = update_second(V, W, H)

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H