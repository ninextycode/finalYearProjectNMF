import numpy as np

def from_WH_to_rank_1_list(W, H):
    l = []
    for i in range(W.shape[1]):
        l.append(W[:, i:i+1] @ H[i:i+1, :])
    return l


def from_rank_1_list_to_WH(rank_1_list):
    cols_W = []
    rows_H = []

    for V in rank_1_list:
        w, h = from_rank_1_matrix_to_wh(V)
        cols_W.append(w)
        rows_H.append(h)

    W = np.hstack(cols_W)
    H = np.vstack(rows_H)
    return W, H


def from_rank_1_matrix_to_wh(V):
    idx = np.where(V > 0)
    idx = [idx[0][0], idx[1][0]]

    v = V[idx[0], idx[1]]

    w = V[:, [idx[1]]]
    h = V[[idx[0]], :] / v

    if np.sum(np.abs(V - w @ h)) / (V.shape[0] * V.shape[1]) > 1e-9:
        raise Exception("Matrix appears to be of rank higher than 1\n{}".format(V))
    return w, h


def rescale_WH(W, H):
    s1 = np.sum(W, axis=0)
    s2 = np.sum(H, axis=1)
    s1[s1 == 0] = 1e-10
    s2[s2 == 0] = 1e-10

    srt_prod = np.sqrt(s1 * s2)
    idx = np.argsort(-1 * srt_prod)
    S1 = (srt_prod / s1).reshape((1, -1))
    S2 = (srt_prod / s2).reshape((-1, 1))
    return (W * S1)[:, idx], (H * S2)[idx, :]
