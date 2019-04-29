# rough work & testing

import numpy as np
import nmf.mult
import nmf.pgrad
import nmf.nesterov
import torch
import nmf_torch.mult
import nmf_torch.pgrad
import nmf_torch.nesterov

import nmf.dtpnn
import nmf.bayes
from theory.visual import InteractiveFactorPlot
from theory.represent import from_WH_to_rank_1_list
import matplotlib.pyplot as plt
from read_data import reading


np.set_printoptions(precision=3, suppress=True)


def plot_algos():
    seed = 1
    np.random.seed(seed)

    inner_dim = 4
    size = (6, 5)
    W_correct = torch.tensor(10 * np.random.rand(size[0], inner_dim))
    H_correct = torch.tensor(np.random.rand(inner_dim, size[1]))

    A_unnoised = W_correct @ H_correct
    A = A_unnoised + 0.1 * torch.tensor(np.random.rand(*size))

    W_init = torch.tensor(np.random.rand(*(A.shape[0], inner_dim)).astype(float))
    H_init = torch.tensor(np.random.rand(*(inner_dim, A.shape[1])).astype(float))

    print("W_init", W_init)
    print("H_init", H_init)

    W, H, errors_mult = nmf_torch.mult.factorize_Fnorm(A, inner_dim, record_errors=True,
                                                       max_steps=10000, epsilon=0,
                                                       W_init=W_init.clone(),
                                                       H_init=H_init.clone())

    W, H, errors_nest = nmf_torch.nesterov.factorize_Fnorm(A, inner_dim, record_errors=True,
                                                           max_steps=100, epsilon=0,
                                                           W_init=W_init.clone(),
                                                           H_init=H_init.clone())

    W, H, errors_proj_sub = nmf_torch.pgrad.factorize_Fnorm_subproblems(A, inner_dim, record_errors=True,
                                                                        max_steps=100, epsilon=0,
                                                                        W_init=W_init.clone(),
                                                                        H_init=H_init.clone())

    errors = [
        errors_mult,
        errors_nest,
        errors_proj_sub]
    labels = [
        "mult",
        "nesterov",
        "projective"]

    plt.figure()
    for err, lbl in zip(errors, labels):
        plt.plot(err[:, 1], np.log(err[:, 0] / (A.shape[0] * A.shape[1])), label=lbl)
    plt.legend()


def bayes_test():
    np.random.seed(3)

    inner_dim = 4
    size = (6, 5)
    W_correct = 10 * np.random.rand(size[0], inner_dim)
    H_correct = np.random.rand(inner_dim, size[1])

    A_unnoised = W_correct @ H_correct
    A = A_unnoised + 0.12 * np.random.rand(*size)

    W_init = np.random.rand(*(A.shape[0], inner_dim)).astype(float)
    H_init = np.random.rand(*(inner_dim, A.shape[1])).astype(float)

    print("W_init", W_init)
    print("H_init", H_init)

    W, H, errors4 = nmf.mult.factorize_Fnorm(A, inner_dim, record_errors=True,
                                             n_steps=10000, epsilon=0,
                                             W_init=W_init.copy(),
                                             H_init=H_init.copy())

    samples_W, samples_H, errors5 = nmf.bayes.factorize_Gibbs(A, inner_dim, record_errors=True,
                                              n_steps=8000,
                                              W_prior=np.ones(W_init.shape),
                                              H_prior=np.ones(H_init.shape))

    W_mean = np.mean(samples_W, axis=0)
    H_mean = np.mean(samples_H, axis=0)

    W = samples_W[-1, :, :]
    H = samples_H[-1, :, :]

    plt.figure()
    plt.subplot(141); plt.imshow(A_unnoised); plt.title("A_unnoised")
    plt.subplot(142); plt.imshow(A); plt.title("A")
    plt.subplot(143); plt.imshow(W_mean @ H_mean); plt.title("W_mean @ H_mean")
    plt.subplot(144); plt.imshow(W @ H); plt.title("W @ H last")


    def get_ci_size(samples, ci_size=80):
        offset = (100 - ci_size) / 2
        A_min = np.percentile(samples, offset, axis=0, interpolation="nearest")
        A_max = np.percentile(samples, 100-offset, axis=0, interpolation="nearest")
        return A_max - A_min

    plt.figure()
    plt.subplot(231)
    plt.imshow(W_correct); plt.title("W_correct")
    plt.subplot(232)
    plt.imshow(W_mean); plt.title("W_mean")
    plt.subplot(233)
    plt.imshow(get_ci_size(samples_W)); plt.title("W_get_ci_size")

    plt.subplot(234)
    plt.imshow(H_correct); plt.title("H_correct");
    plt.subplot(235)
    plt.imshow(H_mean); plt.title("H_mean")
    plt.subplot(236)
    plt.imshow(get_ci_size(samples_H)); plt.title("H_get_ci_size")


    print(W @ H)

    errors = [
        errors4,
        errors5]
    labels = [
        "mult",
        "bayes GIBBS"]

    rank_1_list = from_WH_to_rank_1_list(W, H)


    plt.figure()
    for err, lbl in zip(errors, labels):
        print(err)

    plt.legend()
    


def plot_factorization_test(A, r):
    W, H, errors = nmf.nesterov.factorize_Fnorm(A, r, max_steps=100, epsilon=0,
                                                record_errors=True)
    return InteractiveFactorPlot(W, H, A)


def read_data_reuters():
    data = reading.read_reuters21578("data/reuters21578")
    print(data.shape)


def read_data_indian_pines():
    images = reading.read_pines("data/indian_pines/images")
    ns_line_im = images["site3_im"]
    ori_shape = ns_line_im.shape
    print(ns_line_im.shape)

    plt.figure()
    plt.subplot(161); plt.imshow(ns_line_im[0, :, :])
    plt.subplot(162); plt.imshow(ns_line_im[43, :, :])
    plt.subplot(163); plt.imshow(ns_line_im[87, :, :])
    plt.subplot(164); plt.imshow(ns_line_im[131, :, :])
    plt.subplot(165); plt.imshow(ns_line_im[175, :, :])
    plt.subplot(166); plt.imshow(ns_line_im[219, :, :])

    ns_line_im = ns_line_im.reshape(ori_shape[0], -1)
    print(ns_line_im.shape)

    plt.figure()
    plt.imshow(ns_line_im, aspect="auto")

    ns_line_im = ns_line_im.reshape(*ori_shape)
    print(ns_line_im.shape)

    plt.figure()
    plt.subplot(161); plt.imshow(ns_line_im[0, :, :])
    plt.subplot(162); plt.imshow(ns_line_im[43, :, :])
    plt.subplot(163); plt.imshow(ns_line_im[87, :, :])
    plt.subplot(164); plt.imshow(ns_line_im[131, :, :])
    plt.subplot(165); plt.imshow(ns_line_im[175, :, :])
    plt.subplot(166); plt.imshow(ns_line_im[219, :, :])

    


def read_data_faces():
    ims = reading.read_face_images("data/att_faces/images/")

    imrows = [
        np.hstack([ims[i, :, :] for i in range(a, a+40)])
        for a in range(0, 400, 40)
    ]

    immatrix = np.vstack(imrows)

    plt.figure()
    plt.imshow(immatrix, cmap="gray")

    ims = ims.reshape(ims.shape[0], -1)
    print(ims.shape)

    plt.figure()
    plt.imshow(ims, cmap="gray", aspect="auto")

    

if __name__ == "__main__":
    plot_factorization_test(np.random.rand(10, 10), 5)
    plt.show()