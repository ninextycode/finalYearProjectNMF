def from_WH_to_rank_1_list(W, H):
    l = []
    for i in range(W.shape[1]):
        l.append(W[:, i:i+1] @ H[i:i+1, :])
    return l
