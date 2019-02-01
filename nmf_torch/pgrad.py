import numpy as np
import torch
from nmf_torch.norms import norm_Frobenius
from nmf_torch.mult import update_empty_initials
from time import process_time


def factorise_Fnorm_subproblems(V, inner_dim, n_steps=10000, epsilon=1e-6,
                    record_errors=False, W_init=None, H_init=None):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    start_time = process_time()
    err = float(norm_Frobenius(V - W @ H))
    errors = [(err,  process_time() - start_time)]

    dFWt = dFnorm_H(H @ V.t(), H @ H.t(), W.t())
    dFH = dFnorm_H(W.t() @ V, W.t() @ W, H)

    norm_dFpWt_2 = dH_projected_norm2(dFWt, W.t())
    norm_dFpH_2 = dH_projected_norm2(dFH, H)
    pgrad_norm = torch.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    min_pgrad_main = epsilon * pgrad_norm
    min_pgrad_W = max(1e-3, epsilon) * pgrad_norm
    min_pgrad_H = min_pgrad_W

    for i in range(n_steps):
        if pgrad_norm < min_pgrad_main:
            break

        W, min_pgrad_W, norm_dFpWt_2 = \
            pgd_subproblem_H(V.t(), H.t(), W.t(), min_pgrad_W)
        W = W.t()

        H, min_pgrad_H, norm_dFpH_2 = \
            pgd_subproblem_H(V, W, H, min_pgrad_H)

        err = float(norm_Frobenius(V - W @ H))
        if record_errors:
            errors.append((err, process_time() - start_time))

        pgrad_norm = torch.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H


def pgd_subproblem_H(V, W, H, min_pgrad, n_maxiter=1000):
    H_new = H
    alpha = 1
    WtV = W.t() @ V
    WtW = W.t() @ W

    dF = dFnorm_H(WtV, WtW, H)
    norm_dFpH_2 = dH_projected_norm2(dF, H)
    if torch.sqrt(norm_dFpH_2) < min_pgrad:
        return H_new, min_pgrad / 10, norm_dFpH_2

    for i in range(n_maxiter):
        H_new, alpha = pgd_subproblem_step(WtW, H, dF, alpha)
        H = H_new
        dF = dFnorm_H(WtV, WtW, H)
        norm_dFpH_2 = dH_projected_norm2(dF, H)
        if torch.sqrt(norm_dFpH_2) < min_pgrad:
            break
    return H, min_pgrad, norm_dFpH_2


def pgd_subproblem_step(WtW, H, dF, alpha,
                        beta=0.1, max_alpha_search_iters=10):
    max_a = beta ** -max_alpha_search_iters
    min_a = beta ** max_alpha_search_iters
    alpha = np.clip(alpha, min_a, max_a)

    H_new = project(H - alpha * dF)
    C = pgd_subproblem_step_condition(WtW, H, H_new, dF)

    should_increase = C <= 0

    while max_a >= alpha >= min_a:
        if should_increase:
            alpha = alpha / beta
        else:
            alpha = alpha * beta

        H_prev = H_new
        H_new = project(H - alpha * dF)
        C = pgd_subproblem_step_condition(WtW, H, H_new, dF)

        if should_increase:
            if not C <= 0 or (H_prev == H_new).all():
                alpha = alpha * beta
                H_new = H_prev
                break
        else:
            if C <= 0:
                break

    return H_new, alpha


def pgd_subproblem_step_condition(WtW, H_old, H_new, dF, sigma=0.01):
    d = H_new - H_old
    C = (1 - sigma) * torch.sum(dF * d) + 1/2 * torch.sum(d * (WtW @ d))
    return C


def pgd_subproblem_step_condition_not_simplified(V, W, H_old, H_new, dF, sigma=0.01):
    f_old = 1/2 * torch.sum((V - W @ H_old) ** 2)
    f_new = 1/2 * torch.sum((V - W @ H_new) ** 2)
    d = H_new - H_old
    C = (f_new - f_old) - sigma * torch.sum(dF * d)
    return C

# Fnorm = || V - WH || ^ 2
def dFnorm_H(WtV, WtW, H):
    return WtW @ H - WtV


def dKL_H(V, W, H):
    WH = W @ H
    return ((WH - V) / WH).t() @ W


def dH_projected(dF, H):
    dF = dF.copy()
    dF[H <= 0] = torch.clamp(dF[H <= 0], -np.inf, 0)
    return dF

def dH_projected_norm2(dF, H):
    return torch.sum(dF[(H > 0) | (dF < 0)] ** 2)



def project(A):
    return torch.clamp(A, 0, np.inf)


def factorise_Fnorm_direct(V, inner_dim, n_steps=10000, epsilon=1e-6,
                    record_errors=False, W_init=None, H_init=None):
    W, H = update_empty_initials(V, inner_dim, W_init, H_init)

    # Given any random initial (W, H), very often after
    # the first iteration W^2 = 0 and H^2 = 0 cause the algorithm to stop.
    # The solution (0, 0) is a useless stationary point
    # A simple remedy is to find a new initial point (W1, H1)
    # so that f(W1, H1) < f(0, 0).
    # We can solve it by picking a better initial W and H,
    # one step is enough to get a good enough starting point
    W, H = factorise_Fnorm_subproblems(V, inner_dim, n_steps=1, W_init=W, H_init=H)

    HVt = H @ V.t()
    HHt = H @ H.t()
    WtV = W.t() @ V
    WtW = W.t() @ W

    dFWt = dFnorm_H(HVt, HHt, W.t())
    dFH = dFnorm_H(WtV, WtW, H)
    norm_dFpWt_2 = dH_projected_norm2(dFWt, W.t())
    norm_dFpH_2 = dH_projected_norm2(dFH, H)

    err = float(norm_Frobenius(V - W @ H))
    start_time = process_time()
    errors = [(err, process_time() - start_time)]

    pgrad_norm = torch.sqrt(norm_dFpWt_2 + norm_dFpH_2)
    min_pgrad_main = epsilon * pgrad_norm

    alpha = 1
    for i in range(n_steps):
        if pgrad_norm < min_pgrad_main:
            break

        W, H, alpha = pgd_global_step(V, W, H, dFWt.t(), dFH, alpha)

        err = float(norm_Frobenius(V - W @ H))
        if record_errors:
            errors.append((err, process_time() - start_time))

        WtV = W.t() @ V
        WtW = W.t() @ W
        HVt = H @ V.t()
        HHt = H @ H.t()

        dFWt = dFnorm_H(HVt, HHt, W.t())
        dFH = dFnorm_H(WtV, WtW, H)
        norm_dFpWt_2 = dH_projected_norm2(dFWt, W.t())
        norm_dFpH_2 = dH_projected_norm2(dFH, H)

        pgrad_norm = torch.sqrt(norm_dFpWt_2 + norm_dFpH_2)

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H


def pgd_global_step(V, W, H, dFW, dFH, start_alpha,
                        beta=0.1, max_alpha_search_iters=20):
    max_a = beta ** -max_alpha_search_iters
    min_a = beta ** max_alpha_search_iters
    alpha = np.clip(start_alpha, min_a, max_a)

    W_new, H_new = project(W - alpha * dFW), project(H - alpha * dFH)
    C = pgd_global_step_condition(V, (W, H), (W_new, H_new), dFW, dFH)

    should_increase = C <= 0

    while max_a >= alpha >= min_a:
        if should_increase:
            alpha = alpha / beta
        else:
            alpha = alpha * beta

        W_prev, H_prev = W_new, H_new
        W_new, H_new = project(W - alpha * dFW), project(H - alpha * dFH)
        C = pgd_global_step_condition(V, (W, H), (W_new, H_new), dFW, dFH)

        if should_increase:
            if not C <= 0 or ((H_prev == H_new).all() and (W_prev == W_new).all()):
                alpha = alpha * beta
                H_new = H_prev
                W_new = W_prev
                break
        else:
            if C <= 0:
                break

    return W_new, H_new, alpha


def pgd_global_step_condition(V, WH_old, WH_new, df_W, df_H, sigma=0.01):
    W_old, H_old = WH_old
    W_new, H_new = WH_new

    f_old = 1/2 * torch.sum((V - W_old @ H_old) ** 2)
    f_new = 1/2 * torch.sum((V - W_new @ H_new) ** 2)

    dW = W_new - W_old
    dH = H_new - H_old

    C = (f_new - f_old) - sigma * (torch.sum(df_W * dW) + torch.sum(df_H * dH))
    return C