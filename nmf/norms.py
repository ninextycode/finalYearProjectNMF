import numpy as np


def norm_Frobenius(A, B):
    f = np.sqrt( np.sum((A-B) ** 2))
    return f


def divergence_KullbackLeible(A, B):
    B[B == 0] = 1e-6
    AdivB = A / B
    AdivB[AdivB == 0] = 1e-6
    d = np.sum(A * np.log(AdivB) - A + B)
    return d
