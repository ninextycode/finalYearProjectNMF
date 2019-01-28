import numpy as np
import torch


def norm_Frobenius(A):
    f = torch.sqrt(torch.sum(A ** 2))
    return f


def divergence_KullbackLeible(A, B):
    B[B == 0] = 1e-6
    AdivB = A / B
    AdivB[AdivB == 0] = 1e-6
    d = torch.sum(A * torch.log(AdivB) - A + B)
    return d
