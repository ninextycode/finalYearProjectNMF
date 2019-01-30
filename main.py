import numpy as np
import nmf.mult
import nmf.pgrad
import nmf.nesterov
import nmf.dtpnn
import nmf.bayes
from visualisation.fact import InteractiveFactorPlot
from theory.represent import rescale_WH, from_WH_to_rank_1_list
import matplotlib.pyplot as plt



np.set_printoptions(precision=3, suppress=True)

def plot_algos():
    inner_dim = 40
    size = (60, 50)
    W_correct = 10 * np.random.rand(size[0], inner_dim)
    H_correct = np.random.rand(inner_dim, size[1])

    A_unnoised = W_correct @ H_correct
    A = A_unnoised + 0.1 * np.random.rand(*size)

    W_init = np.random.rand(*(A.shape[0], inner_dim)).astype(float)
    H_init = np.random.rand(*(inner_dim, A.shape[1])).astype(float)

    print("W_init", W_init)
    print("H_init", H_init)

    W, H, errors_mult = nmf.mult.factorise_Fnorm(A, inner_dim, record_errors=True,
                                                 n_steps=10000, min_err=0,
                                                 W_init=W_init.copy(),
                                                 H_init=H_init.copy())

    W, H, errors_nest = nmf.nesterov.factorise_Fnorm(A, inner_dim, record_errors=True,
                                                     n_steps=1000, epsilon=0,
                                                     W_init=W_init.copy(),
                                                     H_init=H_init.copy())

    W, H, errors_proj_sub = nmf.pgrad.factorise_Fnorm_subproblems(A, inner_dim, record_errors=True,
                                                  n_steps=1000, epsilon=0,
                                                  W_init=W_init.copy(),
                                                  H_init=H_init.copy())

    W, H, errors_bayes = nmf.bayes.factorise_ICM(A, inner_dim, record_errors=True,
                                                 n_steps=80000, min_err=0,
                                                 W_init=W_init.copy(),
                                                 H_init=H_init.copy())

    W, H, errors_mult = nmf.mult.factorise_Fnorm(A, inner_dim, record_errors=True,
                                                 n_steps=10000, min_err=0,
                                                 W_init=W_init.copy(),
                                                 H_init=H_init.copy())
    W_mean = np.mean(samples_W, axis=0)
    H_mean = np.mean(samples_H, axis=0)

    W = samples_W[-1, :, :]
    H = samples_H[-1, :, :]

    plt.figure()
    plt.subplot(141);
    plt.imshow(A_unnoised);
    plt.title("A_unnoised")
    plt.subplot(142);
    plt.imshow(A);
    plt.title("A")
    plt.subplot(143);
    plt.imshow(W_mean @ H_mean);
    plt.title("W_mean @ H_mean")
    plt.subplot(144);
    plt.imshow(W @ H);
    plt.title("W @ H last")

    def get_ci_size(samples, ci_size=80):
        offset = (100 - ci_size) / 2
        A_min = np.percentile(samples, offset, axis=0, interpolation="nearest")
        A_max = np.percentile(samples, 100 - offset, axis=0, interpolation="nearest")
        return A_max - A_min

    plt.figure()
    plt.subplot(231)
    plt.imshow(W_correct);
    plt.title("W_correct")
    plt.subplot(232)
    plt.imshow(W_mean);
    plt.title("W_mean")
    plt.subplot(233)
    plt.imshow(get_ci_size(samples_W));
    plt.title("W_get_ci_size")

    plt.subplot(234)
    plt.imshow(H_correct);
    plt.title("H_correct");
    plt.subplot(235)
    plt.imshow(H_mean);
    plt.title("H_mean")
    plt.subplot(236)
    plt.imshow(get_ci_size(samples_H));
    plt.title("H_get_ci_size")

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
        plt.plot(err[:, 1], np.log(err[:, 0] / (A.shape[0] * A.shape[1])), label=lbl)
    plt.legend()
    plt.show()


def bayes_text():
    # np.random.seed(3)

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

    W, H, errors4 = nmf.mult.factorise_Fnorm(A, inner_dim, record_errors=True,
                                             n_steps=10000, min_err=0,
                                             W_init=W_init.copy(),
                                             H_init=H_init.copy())

    samples_W, samples_H, errors5 = nmf.bayes.factorise_Gibbs(A, inner_dim, record_errors=True,
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
    plt.show()





def plot_factorisation_test(A, r):
    W, H, errors = nmf.nesterov.factorise_Fnorm(A, r, n_steps=100, epsilon=0,
                                     record_errors=True)
    return InteractiveFactorPlot(W, H, A)



if __name__ == "__main__":
    # bayes_text()
    A = np.array([
        [0.35, 1, 1, 1, 1],
        [1,    1, 1, 0, 0],
        [0,    0, 1, 1, 0],
        [0,    0, 0, 1, 1],
        [0,    1, 0, 0, 1]
    ])
    r = 10
    A = np.random.rand(r + 5, r) @ np.random.rand(r, r + 10)
    plot = plot_factorisation_test(A, r)
    plt.show()