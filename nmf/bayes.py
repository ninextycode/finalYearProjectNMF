import numpy as np
from nmf.norms import norm_Frobenius
from nmf.mult import update_empty_initials
from time import time as get_time
from nmf.pgrad import project
from scipy.special import erfc, erfcinv, erf, erfinv
from scipy.stats import invgamma
from nmf.pgrad import factorize_Fnorm_subproblems
from itertools import count


# Bayesian NMF algorithm where parameters are estimated by the Iterated Conditional Modes method 
def factorize_ICM(V, inner_dim,
                  max_steps, min_err=0, time_limit=np.inf,
                  record_errors=False,
                  W_prior=None, H_prior=None,
                  shape_prior=0, scale_prior=0):
    W_prior, H_prior = update_empty_initials(V, inner_dim, W_prior, H_prior)
    W = W_prior.copy()
    H = H_prior.copy()

    start_time = get_time()
    err = norm_Frobenius(V - W @ H)
    time = get_time() - start_time
    errors = [(time, err)]

    variance = scale_prior / (1 + shape_prior)
    for i in count():
        if i >= max_steps:
            break
        if err < min_err:
            break
        if time > time_limit:
            break

        HHt = H @ H.T
        VHt = V @ H.T
        W = update_W_ICM(VHt, HHt, W, W_prior, variance)

        err2 = np.sum((V - W @ H) ** 2)
        variance = get_variance_ICM(V.shape, err2, shape_prior, scale_prior)

        WtW = W.T @ W
        VtW = V.T @ W
        H = update_W_ICM(VtW, WtW, H.T, H_prior.T, variance).T

        err = np.sqrt(err2)
        time = get_time() - start_time
        if record_errors:
            errors.append((time, err))

    if record_errors:
        return W, H, np.array(errors)
    else:
        return W, H


def update_W_ICM(VHt, HHt, W, W_prior, variance):
    for n in range(W.shape[1]):
        W[:, n] = (VHt[:, n] - (W @ HHt[:, n] - W[:, [n]] @ HHt[[n], n])
                   - variance / W_prior[:, n]) / max(HHt[n, n], 1e-8)
        W[:, n] = project(W[:, n])

        # avoid zero stationary point
        idx = np.where(W[:, n] == 0)
        W[idx, n] = [np.random.rand() * 1e-6 for i in idx]

    return W


def get_variance_ICM(V_shape, err, shape_prior=0, scale_prior=0):
    return ((err / 2) + scale_prior) /\
           (V_shape[0] * V_shape[1] / 2 + shape_prior + 1)


# Bayesian NMF algorithm where parameters are estimated by the means of Gibbs sampling.  
def factorize_Gibbs(V, inner_dim, n_steps=80000,
                    record_errors=False,
                    W_prior=None, H_prior=None,
                    shape_prior=0, scale_prior=0):
    W_prior, H_prior = update_empty_initials(V, inner_dim, W_prior, H_prior)
    W = W_prior.copy()
    H = H_prior.copy()

    start_time = get_time()
    err = norm_Frobenius(V - W @ H)
    errors = [(get_time() - start_time, err)]

    variance = scale_prior / (1 + shape_prior)

    start_collect_samples = n_steps // 2
    samples_W = np.zeros((n_steps - start_collect_samples, *W.shape))
    samples_H = np.zeros((n_steps - start_collect_samples, *H.shape))

    for i in range(n_steps):
        HHt = H @ H.T
        VHt = V @ H.T
        W = update_W_Gibbs(VHt, HHt, W, W_prior, variance)

        err2 = np.sum((V - W @ H)**2)
        variance = sample_variance_Gibbs(V.shape, err2,
                                         shape_prior,
                                         scale_prior)

        WtW = W.T @ W
        VtW = V.T @ W
        H = update_W_Gibbs(VtW, WtW, H.T, H_prior.T, variance).T

        errors.append((get_time() - start_time, np.sqrt(err2)))

        if i >= start_collect_samples:
            samples_W[i - start_collect_samples, :, :] = W
            samples_H[i - start_collect_samples, :, :] = H
    if record_errors:
        return samples_W, samples_H, np.array(errors)
    else:
        return samples_W, samples_H


def update_W_Gibbs(VHt, HHt, W, W_prior, variance):
    for n in range(W.shape[1]):
        mean = (VHt[:, n]
                - (W @ HHt[:, n] - W[:, [n]] @ HHt[[n], n])\
                - variance / W_prior[:, n]
                ) / max(HHt[n, n], 1e-8)
        W[:, n] = sample_rectified_gaussian_2(mean,
                                            variance / HHt[n, n],
                                            1 / W_prior[:, n])
    return W


def sample_variance_Gibbs(Vshape, err2, shape_prior, scale_prior):
    shape = Vshape[0] * Vshape[1] / 2 + shape_prior + 1
    scale = err2 / 2 + scale_prior
    return invgamma.rvs(a=shape, scale=scale)


# mu - l * s^2 + s * sqrt(2) * InverseErf(y - erf((mu - l s^2)/(s * sqrt(2))) + y * erf( (m - l s^2)/(s * sqrt(2)) ) )
def sample_rectified_gaussian_2(mu, s2, l):
    y = np.random.rand(*mu.shape)
    x = mu - l * s2 + np.sqrt(2 * s2) * erfinv(
        y
        + (y - 1) * erf((mu - l * s2) / (np.sqrt(s2 * 2)))
    )
    x[np.isnan(x) | (x < 0) | np.isinf(x)] = 0
    return x
