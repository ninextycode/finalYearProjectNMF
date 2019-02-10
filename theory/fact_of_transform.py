import numpy as np
from theory.transform import matrix_from_formula, remove_variables
from theory.represent import from_rank_1_list_to_WH
from itertools import zip_longest
import matplotlib.pyplot as plt



def create_factorisation_var_gadgets(num_mat, g1_indices_by_var, solution, fact_data):
    for var, v1, v2 in fact_data["positive"]["s"]["expanded_vars"]:
        solution[var] = solution[v1] * solution[v2]

    for var, v1, v2 in fact_data["negative"]["s"]["expanded_vars"]:
        solution[var] = solution[v1] * solution[v2]

    L = fact_data["negative"]["p"]["var_result"]
    coeffs = fact_data["positive"]["p"]["coeffs"]
    vars = fact_data["positive"]["p"]["vars"]
    solution[L] = 0
    for s, v in zip_longest(coeffs, vars):
        solution[L] += s * solution.get(v, 1)
    terms = []

    shape = num_mat.shape

    for var, g1_idx in g1_indices_by_var.items():
        new_terms = create_factorisation_strong_vg(fact_data["var_counts"][var],
                                                   solution[var], fact_data["ranges"][var],
                                                   shape, g1_idx)
        terms.extend(new_terms)

    s_expanded_vars = fact_data["positive"]["s"]["expanded_vars"] + fact_data["negative"]["s"]["expanded_vars"]
    s_idxs = fact_data["positive"]["s"]["idxs"] + fact_data["negative"]["s"]["idxs"]

    clean_grid = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
    for exp_vars, idx in zip(s_expanded_vars, s_idxs):
        prod, v1, v2 = [solution[e] for e in exp_vars]
        new_terms = create_factorisation_s(v1, v2, shape,
                                           [clean_grid[0][idx], clean_grid[1][idx]])
        terms.extend(new_terms)

    positive_vals = [solution[v] for v in fact_data["positive"]["p"]["vars"]]
    positive_coeffs = fact_data["positive"]["p"]["coeffs"]
    idx = fact_data["positive"]["p"]["idx"]
    coeffs_vals_list = list(zip_longest(positive_coeffs, positive_vals, fillvalue=1))
    new_terms = create_factorisation_p(coeffs_vals_list, shape,  [clean_grid[0][idx], clean_grid[1][idx]])
    terms.extend(new_terms)

    negative_vals = [solution[v] for v in fact_data["negative"]["p"]["vars"]]
    negative_coeffs = fact_data["negative"]["p"]["coeffs"]
    idx = fact_data["negative"]["p"]["idx"]
    coeffs_vals_list = list(zip_longest(negative_coeffs, negative_vals, fillvalue=1))
    new_terms = create_factorisation_p(coeffs_vals_list, shape, [clean_grid[0][idx], clean_grid[1][idx]])
    terms.extend(new_terms)

    return terms


def create_factorisation_strong_vg(num_of_occurencies, val, var_range, shape, g1_idx):
    terms = []
    new_terms = get_top_corner_terms_vg(num_of_occurencies, val, var_range, shape, g1_idx)
    terms.extend(new_terms)

    new_terms = get_middle_block_terms_vg(num_of_occurencies, val, var_range, shape, g1_idx)
    terms.extend(new_terms)

    return np.array(terms)


def get_top_corner_terms_vg(num_of_occ, val, val_range, shape, g1_idx):
    u = 1 / (val_range[1] + 1 - val)
    terms = [np.zeros(shape) for i in range(4)]

    f0_idx = [[[0], [4]], [[0, 1]]]
    terms[0][g1_idx[0][f0_idx], g1_idx[1][f0_idx]] = [[1], [u]]

    f1_idx = [[[1], [4]], [[1, 2]]]
    terms[1][g1_idx[0][f1_idx], g1_idx[1][f1_idx]] = [[1], [1-u]]

    f2_idx = [[[2], [4]], [[2, 3]]]
    terms[2][g1_idx[0][f2_idx], g1_idx[1][f2_idx]] = [[1], [u]]

    f3_idx = [[[3], [4]], [[0, 3] + [i for i in range(4, num_of_occ * 5, 5)]]]
    terms[3][g1_idx[0][f3_idx], g1_idx[1][f3_idx]] = [[1], [1-u]]

    return terms


def get_middle_block_terms_vg(num_of_occurencies, val, var_range, shape, g1_idx):
    terms = []
    for i in range(num_of_occurencies):
        new_terms = get_one_middle_block_factor_vg(num_of_occurencies,
                                                   val, var_range, shape,
                                                   g1_idx, i)
        terms.extend(new_terms)
    return terms


def get_one_middle_block_factor_vg(num_of_occ, val, var_range, shape, g1_idx, block_i):
    block_idx = [None, None]
    block_slice = [slice(5 + 5 * block_i, None), slice(4 + 5 * block_i, None)]
    block_idx[0] = g1_idx[0][block_slice]
    block_idx[1] = g1_idx[1][block_slice]

    u = 1 / (var_range[1] + 1 - val)
    l = (var_range[1] - var_range[0]) / (var_range[1] - var_range[0] + 1)
    a = (1 - u)
    terms = create_top_left_factorisation_simple_vg(a, l, shape, block_idx)
    terms.append(np.zeros(shape))

    offset_for_1s = 5 * (num_of_occ - block_i) + block_i
    terms[-1][block_idx[0][0, 0], block_idx[1][0, 0]] = u
    terms[-1][block_idx[0][0, offset_for_1s], block_idx[1][0, offset_for_1s]] = 1
    terms[-1][block_idx[0][offset_for_1s, 0], block_idx[1][offset_for_1s, 0]] = 1

    terms[-1][block_idx[0][offset_for_1s, offset_for_1s],
              block_idx[1][offset_for_1s, offset_for_1s]] = 1 / u

    top_row = 4
    terms[-1][g1_idx[0][[top_row], [4 + block_i * 5, 4 + 5 * num_of_occ + block_i]],
              g1_idx[1][[top_row], [4 + block_i * 5, 4 + 5 * num_of_occ + block_i]]] = [[u, 1]]

    return terms


def create_top_left_factorisation_simple_vg(a, l, shape, idx):
    terms = [np.zeros(shape) for i in range(4)]

    terms[0][idx[0][:2, :5], idx[1][:2, :5]] = np.array([
        [a, a, a, 0, 0],
        [l, l, l, 0, 0]
    ])
    terms[1][idx[0][[0, 2], :5], idx[1][[0, 2], :5]] = np.array([
        [0, 0, l - a, l - a, 0],

        [0, 0, l, l, 0]
    ])
    terms[2][idx[0][[0, 3], :5], idx[1][[0, 3], :5]] = np.array([
        [0, 0, 0, a, a],

        [0, 0, 0, l, l]
    ])
    terms[3][idx[0][[0, 4], :5], idx[1][[0, 4], :5]] = np.array([
        [0, l - a, 0, 0, l - a],

        [0, l, 0, 0, l]
    ])
    return terms


def create_factorisation_s(val1, val2, shape, idx):
    terms = []
    idx_block_1 = [idx[0][:5, 6:], idx[1][:5, 6:]]
    idx_block_2 = [idx[0][6:, :5], idx[1][6:, :5]]

    terms_vg_1 = create_top_left_factorisation_simple_vg(1 - val1, 1, shape, idx_block_1)
    terms_vg_2 = create_top_left_factorisation_simple_vg(1 - val2, 1, shape, idx_block_2)

    terms.extend(terms_vg_1)
    terms.extend(terms_vg_2)
    terms.append(np.zeros(shape))

    terms[-1][idx[0][5:7, 5:7], idx[1][5:7, 5:7]] = 1
    terms[-1][idx[0][0, 0], idx[1][0, 0]] = val1 * val2
    terms[-1][idx[0][0, [5, 6]], idx[1][0, [5, 6]]] = val1
    terms[-1][idx[0][[5, 6], 0], idx[1][[5, 6], 0]] = val2

    return terms


def create_factorisation_p(coeffs_vals_list, shape, idx):
    t = len(coeffs_vals_list)
    sum_val = 0
    terms = []
    # Note that empty list will result in no factors, which is correct
    # as in this case the corresponding P matrix is just one zero
    for i, (s, v) in enumerate(coeffs_vals_list):
        sum_val += s * v
        block_idx = [slice(t + 1 + i * 5, t + 1 + (i + 1) * 5),
                    [0] + list(range(1 + 4 * i, 1 + 4 * (i + 1)))]

        idx_i = [idx[0][block_idx],
                 idx[1][block_idx]]
        new_terms = create_top_left_factorisation_simple_vg(1 - v, 1, shape, idx_i)
        terms.extend(new_terms)
        terms.append(np.zeros(shape))

        block_idx = [
            [[i], [t], [t + 1 + 5 * i]],
            [[0, 1 + 4 * t + i]]
        ]
        idx_i = [idx[0][block_idx],
                 idx[1][block_idx]]
        terms[-1][idx_i] = [
            [v,     1],
            [v * s, s],
            [v,     1]
        ]

    return terms


def test(formula, solution):
    from theory.transform import plot

    num_mat, var_mat, expected_rank, fact_data = matrix_from_formula(formula)
    print(fact_data)

    N, V, expected_rank, g1_indices_by_var = remove_variables(num_mat, var_mat, fact_data["ranges"], expected_rank)
    plot(N, V)

    plt.figure()
    plt.gca().set_title("small")
    plt.imshow(N)
    plt.gca().set_title("big")

    plt.show(block=False)

    terms = create_factorisation_var_gadgets(N, g1_indices_by_var, solution, fact_data)
    plt.figure()
    plt.imshow(np.sum(terms, axis=0))
    plt.gca().set_title("sum")

    diff = np.sum(np.abs(N - np.sum(terms, axis=0)))
    print("diff", diff)

    W, H = from_rank_1_list_to_WH(terms)

    plt.figure()
    plt.subplot(121)
    plt.imshow(W)
    plt.gca().set_title("W")
    plt.subplot(122)
    plt.imshow(H)
    plt.gca().set_title("H")

    print("expected_rank", expected_rank)
    print("len(terms)", len(terms))



def test_testricted_factorisation(formula):
    import numpy as np
    from nmf.norms import norm_Frobenius, divergence_KullbackLeible
    from nmf.pgrad import project, dFnorm_H, dH_projected_norm2
    from time import time as get_time
    from nmf.mult import update_empty_initials
    from itertools import count

    def factorise_Fnorm(V, inner_dim,
                        max_steps, epsilon=0, time_limit=np.inf,
                        record_errors=False, W_init=None, H_init=None):
        W, H = update_empty_initials(V, inner_dim, W_init, H_init)

        W_mask = W == 0
        H_mask = H == 0

        err = norm_Frobenius(V - W @ H)
        start_time = get_time()
        time = get_time() - start_time
        errors = [(time, err)]

        dFWt = dFnorm_H(H @ V.T, H @ H.T, W.T)
        dFH = dFnorm_H(W.T @ V, W.T @ W, H)
        norm_dFpWt_2 = dH_projected_norm2(dFWt, W.T)
        norm_dFpH_2 = dH_projected_norm2(dFH, H)
        pgrad_norm = np.sqrt(norm_dFpWt_2 + norm_dFpH_2)

        min_pgrad_main = epsilon * pgrad_norm
        min_pgrad_W = max(1e-3, epsilon) * pgrad_norm
        min_pgrad_H = min_pgrad_W

        for i in count():
            if i >= max_steps:
                break
            if pgrad_norm < min_pgrad_main:
                break
            if time > time_limit:
                break

            W, min_pgrad_W, norm_dFpWt_2 = \
                nesterov_subproblem_H(V.T, H.T, W.T, min_pgrad_W)
            W = W.T
            W[W_mask] = 0
            H, min_pgrad_H, norm_dFpH_2 = \
                nesterov_subproblem_H(V, W, H, min_pgrad_H)
            H[H_mask] = 0
            err = norm_Frobenius(V - W @ H)
            time = get_time() - start_time
            if record_errors:
                errors.append((time, err))

            pgrad_norm = np.sqrt(norm_dFpWt_2 + norm_dFpH_2)

        if record_errors:
            return W, H, np.array(errors)
        else:
            return W, H

    def nesterov_subproblem_H(V, W, H, min_pgrad, n_maxiter=1000):
        a = 1
        WtW = W.T @ W
        WtV = W.T @ V

        dFH = dFnorm_H(WtV, WtW, H)
        norm_dFpH_2 = dH_projected_norm2(dFH, H)
        if np.sqrt(norm_dFpH_2) < min_pgrad:
            return H, min_pgrad / 10, norm_dFpH_2

        L = np.linalg.norm(WtW, ord=2)
        Y = H.copy()
        for i in range(n_maxiter):
            H_next = project(Y - 1 / L * dFnorm_H(WtV, WtW, Y))
            a_next = (1 + np.sqrt(4 * (a ** 2) + 1)) / 2
            Y = H_next + (a - 1) / a_next * (H_next - H)

            a = a_next
            H = H_next

            dFH = dFnorm_H(WtV, WtW, H)
            norm_dFpH_2 = dH_projected_norm2(dFH, H)
            if np.sqrt(norm_dFpH_2) < min_pgrad:
                break

        return H, min_pgrad, norm_dFpH_2

    num_mat, var_mat, expected_rank, fact_data = matrix_from_formula(formula)
    N, V, expected_rank, g1_indices_by_var = remove_variables(num_mat, var_mat, fact_data["ranges"], expected_rank)

    solution = {
        v: (r[0] + r[1]) / 2  for v, r in fact_data["ranges"].items()
    }

    terms = create_factorisation_var_gadgets(N, g1_indices_by_var, solution, fact_data)
    W, H = from_rank_1_list_to_WH(terms)

    W_, H_, errors = factorise_Fnorm(N, time_limit=180, W_init=W, H_init=H,
                                     record_errors=True,
                                     inner_dim=expected_rank,
                                     max_steps=np.inf, epsilon=0)

    diff = np.sum(np.abs(N - W_ @ H_))
    print("diff " + formula, diff)


    plt.plot(errors[:, 0], np.log(errors[:, 1] / (N.shape[0] * N.shape[1])))
    plt.gca().set_title(formula)

    plt.figure()
    plt.imshow(W_ @ H_ - N)
    plt.gca().set_title("diff " + formula)




if __name__ == "__main__":
    formula = "x * y + 2 * z + 2 *  z - 3 * u * v *  t - 4 * z - 5 * u * x + 1"
    solution = dict(
        x=0.5,
        y=0.6,
        z=0.5,
        v=0.5,
        u=0.4,
        t=0.5
    )
    #test(formula, solution)

    test_testricted_factorisation("100 * x * y + 100 * x * z - 250")
    test_testricted_factorisation("100 * x * y + 100 * x * z - 201")
    test_testricted_factorisation("100 * x * y + 100 * x * z - 200")
    test_testricted_factorisation("100 * x * y + 100 * x * z - 100")
    test_testricted_factorisation("100 * x * y + 100 * x * z - 0")

    plt.show()