import numpy as np
from nmf.norms import norm_Frobenius
from nmf.pgrad import project, dFnorm_H, dH_projected_norm2, pgd_subproblem_step_condition
from nmf.mult import update_empty_initials
from time import time as get_time


def factorize_Fnorm(V, inner_dim, n_steps=10000, epsiolon=1e-6,
                    record_errors=False, W_init=None, H_init=None):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    dFWt = dFnorm_H(H @ V.T, H @ H.T, W.T)
    dFH = dFnorm_H(W.T @ V, W.T @ W, H)
    norm_dFpWt_2 = dH_projected_norm2(dFWt, W.T)
    norm_dFpH_2 = dH_projected_norm2(dFH, H)
    pgrad_norm = np.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    min_pgrad_main = epsiolon * pgrad_norm
    min_pgrad_W = max(1e-3, epsiolon) * pgrad_norm
    min_pgrad_H = min_pgrad_W

    start_time = get_time()
    err = norm_Frobenius(V - W @ H)
    errors = [(get_time() - start_time, err)]

    for i in range(n_steps):
        if pgrad_norm < min_pgrad_main:
            break

        W, l_W, min_pgrad_W, norm_dFpWt_2 = \
            dtpnn_subproblem_H(V.T, H.T, W.T, min_pgrad_H)
        W = W.T

        H, l_H, min_pgrad_H, norm_dFpH_2 = \
            dtpnn_subproblem_H(V, W, H, min_pgrad_W)

        err = norm_Frobenius(V - W @ H)
        if record_errors:
            errors.append((get_time() - start_time, err))

        pgrad_norm = np.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H


def dtpnn_subproblem_H(V, W, H, min_pgrad, start_lambda=1):
    l = start_lambda
    WtV = W.T @ V
    WtW = W.T @ W

    dF = dFnorm_H(WtV, WtW, H)

    H_new, l = dtpnn_subproblem_step(WtW, H, dF, l)
    H = H_new
    dF = dFnorm_H(WtV, WtW, H)
    norm_dFpH_2 = dH_projected_norm2(dF, H)

    return H, l, min_pgrad, norm_dFpH_2


def dtpnn_subproblem_step(WtW, H, dF, start_lambda, beta=0.5, max_lambda_search_iters=30):
    alpha = 0.01

    max_l = beta ** -max_lambda_search_iters
    min_l = beta ** max_lambda_search_iters
    l = np.clip(start_lambda, min_l, max_l)

    H_new = next_value(H, dF, l)
    C = pgd_subproblem_step_condition(WtW, H, H_new, dF, l * alpha)
    should_increase = C <= 0

    while max_l >= l >= min_l:
        if should_increase:
            l = l / beta
        else:
            l = l * beta

        H_prev = H_new
        H_new = next_value(H, dF, l)
        C = pgd_subproblem_step_condition(WtW, H, H_new, dF, l * alpha)

        if should_increase:
            if not C <= 0 or (H_prev == H_new).all():
                l = l * beta
                H_new = H_prev
                break
        else:
            if C <= 0:
                break

    return H_new, l

def next_value(H, dFH, l):
    return H + l * (-H + project(H - dFH))