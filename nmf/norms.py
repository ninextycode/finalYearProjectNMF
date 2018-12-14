import numpy as np


def norm_Frobenius(A, B):
    f = np.sqrt( np.sum((A-B) ** 2))
    return f


def divergence_KullbackLeible(A, B):
    d = np.sum(A * np.log(A / B) - A + B)
    return d
