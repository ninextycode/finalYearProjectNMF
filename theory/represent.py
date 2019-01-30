import numpy as np

def from_WH_to_rank_1_list(W, H):
    l = []
    for i in range(W.shape[1]):
        l.append(W[:, i:i+1] @ H[i:i+1, :])
    return l


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
