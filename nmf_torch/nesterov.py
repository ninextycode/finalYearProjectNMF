import numpy as np
from nmf_torch.norms import norm_Frobenius, divergence_KullbackLeible
from nmf_torch.pgrad import project, dFnorm_H, dH_projected_norm2
from time import time as get_time
from nmf_torch.mult import update_empty_initials
import torch
from itertools import count


def factorize_Fnorm(V, inner_dim,
                    max_steps, epsilon=0, time_limit=np.inf,
                    record_errors=False,
                    W_init=None, H_init=None):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    err = float(norm_Frobenius(V - W @ H))
    start_time = get_time()
    time = get_time() - start_time
    errors = [(time, err)]

    dFWt = dFnorm_H(H @ V.t(), H @ H.t(), W.t())
    dFH = dFnorm_H(W.t() @ V, W.t() @ W, H)
    norm_dFpWt_2 = dH_projected_norm2(dFWt, W.t())
    norm_dFpH_2 = dH_projected_norm2(dFH, H)
    pgrad_norm = torch.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    min_pgrad_main = epsilon * pgrad_norm
    min_pgrad_W = max(1e-3, epsilon) * pgrad_norm
    min_pgrad_H = min_pgrad_W

    for i in count():
        if i >= max_steps:
            break
        if pgrad_norm < min_pgrad_main:
            break
        if time > time_limit:
            break

        W, min_pgrad_W, norm_dFpWt_2 = \
            nesterov_subproblem_H(V.t(), H.t(), W.t(), min_pgrad_W)
        W = W.t()

        H, min_pgrad_H, norm_dFpH_2 = \
            nesterov_subproblem_H(V, W, H, min_pgrad_H)

        err = float(norm_Frobenius(V - W @ H))
        time = get_time() - start_time
        if record_errors:
            errors.append((time, err))

        pgrad_norm = torch.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H


def nesterov_subproblem_H(V, W, H, min_pgrad, n_maxiter=1000):
    a = 1
    WtW = W.t() @ W
    WtV = W.t() @ V

    dFH = dFnorm_H(WtV, WtW, H)
    norm_dFpH_2 = dH_projected_norm2(dFH, H)
    if torch.sqrt(norm_dFpH_2) < min_pgrad:
        return H, min_pgrad / 10, norm_dFpH_2

    L = torch.norm(WtW, p=2)
    Y = H.clone()
    for i in range(n_maxiter):
        H_next = project(Y - 1/L * dFnorm_H(WtV, WtW, Y))
        a_next = (1 + np.sqrt(4 * (a ** 2) + 1)) / 2
        Y = H_next + (a - 1) / a_next * (H_next - H)

        a = a_next
        H = H_next

        dFH = dFnorm_H(WtV, WtW, H)
        norm_dFpH_2 = dH_projected_norm2(dFH, H)
        if torch.sqrt(norm_dFpH_2) < min_pgrad:
            break

    return H, min_pgrad, norm_dFpH_2
