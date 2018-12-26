import numpy as np
from nmf.norms import norm_Frobenius

# print = lambda *X: None

def factorise_Fnorm_subproblems_pgrad(V, inner_dim, n_steps=10000, epsiolon=1e-6,
                    record_errors=False, W_init=None, H_init=None):

    alpha_H = 1
    alpha_W = 1

    if W_init is None:
        W = 1 - np.random.rand(V.shape[0], inner_dim)
    else:
        W = W_init

    if H_init is None:
        H = 1 - np.random.rand(inner_dim, V.shape[1])
    else:
        H = H_init

    err = norm_Frobenius(V - W @ H)
    errors = [err]

    dFWt = dFnorm_H(H @ V.T, H @ H.T, W.T)
    dFH = dFnorm_H(W.T @ V, W.T @ W, H)
    dFpWt = dFnorm_H_projected(dFWt, W)
    dFpH = dFnorm_H_projected(dFH, H)

    pgrad_norm = norm_Frobenius(np.hstack([dFpWt, dFpH]))
    min_pgrad_main = epsiolon * pgrad_norm
    min_pgrad_W = max(1e-3, epsiolon) * pgrad_norm
    min_pgrad_H = min_pgrad_W
    for i in range(n_steps):
        if pgrad_norm < min_pgrad_main:
            break

        W, alpha_W, min_pgrad_W, dFpWt = \
            pgd_subproblem_H(V.T, H.T, W.T, alpha_W, min_pgrad_W)
        W = W.T

        H, alpha_H, min_pgrad_H, dFpH = \
            pgd_subproblem_H(V, W, H, alpha_H, min_pgrad_H)

        err = norm_Frobenius(V - W @ H)
        if record_errors:
            errors.append(err)

        pgrad_norm = norm_Frobenius(np.hstack([dFpWt, dFpH]))

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H


def pgd_subproblem_H(V, W, H, start_alpha, min_pgrad, n_maxiter=1000):
    H_new = H
    alpha = start_alpha
    WtV = W.T @ V
    WtW = W.T @ W

    dF = dFnorm_H(WtV, WtW, H)
    dFpH = dFnorm_H_projected(dF, H)
    if norm_Frobenius(dFpH) < min_pgrad:
        return H_new, alpha, min_pgrad / 10, dFpH

    for i in range(n_maxiter):
        H_new, alpha = pgd_subproblem_step(WtV, WtW, H, dF, alpha)
        H = H_new
        dF = dFnorm_H(WtV, WtW, H)
        dFpH = dFnorm_H_projected(dF, H)
        if norm_Frobenius(dFpH) < min_pgrad:
            break
    return H_new, alpha, min_pgrad, dFpH


def pgd_subproblem_step(WtV, WtW, H, dF, alpha,
                        beta=0.1, max_alpha_search_iters=10):
    H_new = project(H - alpha * dF)
    C = pgd_subproblem_step_condition(WtW, H, H_new, dF)

    should_increase = C <= 0
    max_a = beta ** -max_alpha_search_iters
    min_a = beta ** max_alpha_search_iters
    alpha = np.clip(alpha, min_a, max_a)

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
    C = (1 - sigma) * np.sum(dF * d) + 1/2 * np.sum(d * (WtW @ d))
    return C


# Fnorm = || V - WH || ^ 2
def dFnorm_H(WtV, WtW, H):
    return WtW @ H - WtV


def dFnorm_H_projected(dF, H):
    dF = dF.copy()
    dF[H <= 0] = np.clip(dF[H <= 0], -np.inf, 0)
    return dF


def project(A):
    A = A.copy()
    return np.clip(A, 0, np.inf)

def factorise_Fnorm_direct_pgrad(V, inner_dim, n_steps=10000, epsiolon=1e-6,
                    record_errors=False, W_init=None, H_init=None):
    if W_init is None:
        W = 1 - np.random.rand(V.shape[0], inner_dim)
    else:
        W = W_init

    if H_init is None:
        H = 1 - np.random.rand(inner_dim, V.shape[1])
    else:
        H = H_init

    err = norm_Frobenius(V - W @ H)
    errors = [err]

    HVt = H @ V.T
    HHt = H @ H.T
    WtV = W.T @ V
    WtW = W.T @ W

    dFWt = dFnorm_H(HVt, HHt, W.T)
    dFH = dFnorm_H(WtV, WtW, H)
    dFpWt = dFnorm_H_projected(dFWt, W)
    dFpH = dFnorm_H_projected(dFH, H)

    pgrad_norm = norm_Frobenius(np.hstack([dFpWt, dFpH]))
    min_pgrad_main = epsiolon * pgrad_norm

    alpha = 1
    for i in range(n_steps):
        if pgrad_norm < min_pgrad_main:
            break

        W, H, alpha = pgd_global_step(V, W, H, dFWt.T, dFH, alpha)
        print(alpha)
        err = norm_Frobenius(V - W @ H)
        if record_errors:
            errors.append(err)

        WtV = W.T @ V
        WtW = W.T @ W
        HVt = H @ V.T
        HHt = H @ H.T

        dFWt = dFnorm_H(HVt, HHt, W.T)
        dFH = dFnorm_H(WtV, WtW, H)
        dFpWt = dFnorm_H_projected(dFWt, W.T)
        dFpH = dFnorm_H_projected(dFH, H)

        pgrad_norm = norm_Frobenius(np.hstack([dFpWt, dFpH]))

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
    print("C", C)

    should_increase = C <= 0

    while max_a >= alpha >= min_a:
        if should_increase:
            alpha = alpha / beta
        else:
            alpha = alpha * beta

        W_prev, H_prev = W_new, H_new
        W_new, H_new = project(W - alpha * dFW), project(H - alpha * dFH)
        C = pgd_global_step_condition(V, (W, H), (W_new, H_new), dFW, dFH)
        print("C", C)

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

    f_old = norm_Frobenius(V - W_old @ H_old)
    f_new = norm_Frobenius(V - W_new @ H_new)

    dW = W_new - W_old
    dH = H_new - H_old

    C = (f_new - f_old) - sigma * (np.sum(df_W * dW) + np.sum(df_H * dH))
    return C
