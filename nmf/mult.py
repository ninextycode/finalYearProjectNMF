import nmf.norms
import numpy as np
from time import process_time

def update_empty_initials(V, inner_dim, W_init, H_init):
    if W_init is None:
        W = 1 - np.random.rand(V.shape[0], inner_dim)
    else:
        W = W_init

    if H_init is None:
        H = 1 - np.random.rand(inner_dim, V.shape[1])
    else:
        H = H_init
    return W, H

def factorise_Fnorm(V, inner_dim, n_steps=10000, min_err=1e-6,
                    record_errors=False, W_init=None, H_init=None):
    return factorise(V, inner_dim, n_steps, min_err, record_errors,
                     update_first=update_first_Fnorm,
                     update_second=update_second_Fnorm,
                     error=lambda A, B: nmf.norms.norm_Frobenius(A-B),
                     W_init=W_init,
                     H_init=H_init)


def factorise_KLdiv(V, inner_dim, n_steps=10000, min_err=1e-6, record_errors=False,
                    W_init=None, H_init=None):
    return factorise(V, inner_dim, n_steps, min_err, record_errors,
                     update_first=update_first_KLdiv,
                     update_second=update_second_KLdiv,
                     error=nmf.norms.divergence_KullbackLeible,
                     W_init=W_init,
                     H_init=H_init)

def update_first_Fnorm(V, W, H):
    VHt = V  @ H.T
    WHHt = W @ (H @ H.T)

    W = W * VHt / WHHt

    return W


def update_second_Fnorm(V, W, H):
    WtV = W.T @ V

    WtWH = W.T @ W @ H

    H = H * WtV / WtWH

    return H


def update_first_KLdiv(V, W, H):
    WH = W @ H
    WH[WH == 0] = 1e-10
    num = (V / WH) @ H.T
    S_H = np.sum(H, keepdims=True, axis=1).T
    W = W * num / S_H
    return W


def update_second_KLdiv(V, W, H):
    WH = W @ H
    WH[WH==0] = 1e-10
    num = W.T @ (V / WH)
    S_W = np.sum(W, keepdims=True, axis=0).T
    H = H * num / S_W
    return H


def factorise(V, inner_dim, n_steps, min_err, record_errors,
              update_first, update_second, error, W_init, H_init):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    start_time = process_time()
    err = error(V, W @ H)
    errors = [(err, process_time() - start_time)]

    for i in range(n_steps):
        if err < min_err:
            break

        W = update_first(V, W, H)
        # H = update_second(V, W, H)
        H = update_first(V.T, H.T, W.T).T

        err = error(V, W @ H)
        if record_errors:
            errors.append((err,  process_time() - start_time))


    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H