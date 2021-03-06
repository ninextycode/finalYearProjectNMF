import nmf.norms
import numpy as np
from time import time as get_time
from itertools import count


# a piece of code which does nothing if the starting point for the factorization algorithm is provided
# if the starting point is not provided, initial factorization is generated randomly
def update_empty_initials(V, inner_dim, W_init, H_init):
    if W_init is None:
        W = 1 - np.random.rand(V.shape[0], inner_dim)
        W = W.astype(V.dtype)
    else:
        W = W_init

    if H_init is None:
        H = 1 - np.random.rand(inner_dim, V.shape[1])
        H = H.astype(V.dtype)
    else:
        H = H_init
    return W, H

# the multiplicative NMF algorithm which aims to minimise Frobenius norm of the difference
# between the original data and the data reconstructed from the factorization  
def factorize_Fnorm(V, inner_dim,
                    max_steps, epsilon=0, time_limit=np.inf,
                    record_errors=False,
                    W_init=None, H_init=None):
    return factorize(V=V,
                     inner_dim=inner_dim,
                     record_errors=record_errors,
                     update_first=update_first_Fnorm,
                     error=lambda A, B: nmf.norms.norm_Frobenius(A-B),
                     W_init=W_init,
                     H_init=H_init,
                     max_steps=max_steps,
                     epsilon=epsilon,
                     time_limit=time_limit)

# the  multiplicative algorithm which aims to minimise Kullback–Leibler divergence between
# the original data and the data reconstructed from the factorization 
def factorize_KLdiv(V, inner_dim,
                    max_steps, epsilon=0, time_limit=np.inf,
                    record_errors=False,
                    W_init=None, H_init=None):
    return factorize(V=V,
                     inner_dim=inner_dim,
                     record_errors=record_errors,
                     update_first=update_first_KLdiv,
                     error=nmf.norms.divergence_KullbackLeible,
                     W_init=W_init,
                     H_init=H_init,
                     max_steps=max_steps,
                     epsilon=epsilon,
                     time_limit=time_limit)

def update_first_Fnorm(V, W, H):
    VHt = V  @ H.T
    WHHt = W @ (H @ H.T)

    WHHt[WHHt == 0] = 1e-10

    W = W * VHt / WHHt

    return W


def update_second_Fnorm(V, W, H):
    WtV = W.T @ V
    WtWH = W.T @ W @ H
    WtWH[WtWH == 0] = 1e-10
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


def factorize(V, inner_dim, record_errors,
              update_first, error, W_init, H_init,
              max_steps, epsilon, time_limit=np.inf):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    err = error(V, W @ H)
    start_time = get_time()
    time = get_time() - start_time
    errors = [(time, err)]

    for i in count():
        if i >= max_steps:
            break
        if err < epsilon:
            break
        if time > time_limit:
            break

        W = update_first(V, W, H)
        # H = update_second(V, W, H)
        H = update_first(V.T, H.T, W.T).T

        err = error(V, W @ H)
        time = get_time() - start_time
        if record_errors:
            errors.append((time, err))

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H