import nmf_torch.norms
import numpy as np
from time import process_time
from itertools import count
import torch


def update_empty_initials(V, inner_dim, W_init, H_init):
    if W_init is None:
        W = 1 - torch.rand(V.shape[0], inner_dim)
        W = W.to(dtype=V.dtype, device=V.device)
    else:
        W = W_init

    if H_init is None:
        H = 1 - torch.rand(inner_dim, V.shape[1])
        H = H.to(dtype=V.dtype, device=V.device)
    else:
        H = H_init
    return W, H


def factorise_Fnorm(V, inner_dim,
                    max_steps, epsilon=0, time_limit=np.inf,
                    record_errors=False,
                    W_init=None, H_init=None):
    return factorise(V=V,
                     inner_dim=inner_dim,
                     record_errors=record_errors,
                     update_first=update_first_Fnorm,
                     error=lambda A, B: nmf_torch.norms.norm_Frobenius(A-B),
                     W_init=W_init,
                     H_init=H_init,
                     max_steps=max_steps,
                     epsilon=epsilon,
                     time_limit=time_limit)


def factorise_KLdiv(V, inner_dim,
                    max_steps, epsilon=0, time_limit=np.inf,
                    record_errors=False,
                    W_init=None, H_init=None):
    return factorise(V=V,
                     inner_dim=inner_dim,
                     record_errors=record_errors,
                     update_first=update_first_KLdiv,
                     error=nmf_torch.norms.divergence_KullbackLeible,
                     W_init=W_init,
                     H_init=H_init,
                     max_steps=max_steps,
                     epsilon=epsilon,
                     time_limit=time_limit)


def update_first_Fnorm(V, W, H):
    VHt = V  @ H.t()
    WHHt = W @ (H @ H.t())
    WHHt[WHHt == 0] = 1e-10
    W = W * VHt / WHHt
    return W


def update_second_Fnorm(V, W, H):
    WtV = W.t() @ V
    WtWH = W.t() @ W @ H
    WtWH[WtWH == 0] = 1e-10
    H = H * WtV / WtWH
    return H


def update_first_KLdiv(V, W, H):
    WH = W @ H
    WH[WH == 0] = 1e-10
    num = (V / WH) @ H.t()
    S_H = torch.sum(H, keepdim=True, dim=1).t()
    W = W * num / S_H
    return W


def update_second_KLdiv(V, W, H):
    WH = W @ H
    WH[WH==0] = 1e-10
    num = W.t() @ (V / WH)
    S_W = torch.sum(W, keepdim=True, dim=0).t()
    H = H * num / S_W
    return H


def factorise(V, inner_dim, record_errors,
              update_first, error, W_init, H_init,
              max_steps, epsilon, time_limit=np.inf):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    err = float(error(V, W @ H))
    start_time = process_time()
    time = process_time() - start_time
    errors = [(time, err)]

    for i in count():
        if i >= max_steps:
            break
        if err < epsilon:
            break
        if time > time_limit:
            break

        W = update_first(V, W, H)
        H = update_first(V.t(), H.t(), W.t()).t()

        err = float(error(V, W @ H))
        time = process_time() - start_time

        if record_errors:
            errors.append((time, err))

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H
