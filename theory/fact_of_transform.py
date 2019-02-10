import numpy as np
from theory.transform import matrix_from_formula, remove_variables
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
        new_terms = create_factorisation_var_gadget(solution[var], fact_data["ranges"][var],
                                                    shape, g1_idx)
        terms.extend(new_terms)

    s_expanded_vars = fact_data["positive"]["s"]["expanded_vars"] + fact_data["negative"]["s"]["expanded_vars"]
    s_idxs = fact_data["positive"]["s"]["idxs"] + fact_data["negative"]["s"]["idxs"]

    clean_grid = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
    for exp_vars, idx in zip(s_expanded_vars, s_idxs):
        prod, v1, v2 = [solution[e] for e in exp_vars]
        new_terms = create_factorisation_s(v1, v2, shape,
                                           [clean_grid[0][idx], clean_grid[1][idx]])
        print(1)
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


def create_factorisation_var_gadget(val, var_range, shape, g1_idx):
    terms = []
    new_terms = get_top_corner_terms_vg(val, var_range, shape, g1_idx)
    terms.extend(new_terms)

    new_terms = get_middle_block_terms_vg(val, var_range, shape, g1_idx)
    terms.extend(new_terms)

    return terms


def get_top_corner_terms_vg(val, val_range, shape, g1_idx):
    u = 1 / (val_range[1] + 1 - val)
    terms = [np.zeros(shape) for i in range(4)]

    f0_idx = [[[0], [4]], [[0, 1]]]
    terms[0][g1_idx[0][f0_idx], g1_idx[1][f0_idx]] = [[1], [u]]

    f1_idx = [[[1], [4]], [[1, 2]]]
    terms[1][g1_idx[0][f1_idx], g1_idx[1][f1_idx]] = [[1], [1-u]]

    f2_idx = [[[2], [4]], [[2, 3]]]
    terms[2][g1_idx[0][f2_idx], g1_idx[1][f2_idx]] = [[1], [u]]

    n_cols = g1_idx[0].shape[1]
    f3_idx = [[[3], [4]], [[0, 3] + [i for i in range(4, n_cols, 5)]]]
    terms[3][g1_idx[0][f3_idx], g1_idx[1][f3_idx]] = [[1], [1-u]]

    return terms


def get_middle_block_terms_vg(val, var_range, shape, g1_idx):
    t = (g1_idx[0].shape[0] - 5) // 5
    terms = []
    for i in range(t):
        new_terms = get_one_middle_block_factor_vg(val, var_range, shape, g1_idx, i, t)
        terms.extend(new_terms)
    return terms


def get_one_middle_block_factor_vg(val, var_range, shape, g1_idx, block_i, t):
    block_idx = [None, None]
    block_slice = [slice(5 + 5 * block_i, None), slice(4 + 5 * block_i, None)]
    block_idx[0] = g1_idx[0][block_slice]
    block_idx[1] = g1_idx[1][block_slice]

    u = 1 / (var_range[1] + 1 - val)
    l = (var_range[1] - var_range[0]) / (var_range[1] - var_range[0] + 1)
    a = (1 - u)
    terms = create_top_left_factorisation_simple_vg(a, l, shape, block_idx)
    terms.append(np.zeros(shape))

    offset_for_1s = 5 * (t - block_i) + block_i
    terms[4][block_idx[0][0, 0], block_idx[1][0, 0]] = u
    terms[4][block_idx[0][0, offset_for_1s], block_idx[1][0, offset_for_1s]] = 1
    terms[4][block_idx[0][offset_for_1s, 0], block_idx[1][offset_for_1s, 0]] = 1
    terms[4][block_idx[0][offset_for_1s, offset_for_1s],
             block_idx[1][offset_for_1s, offset_for_1s]] = 1 / u
    top_row = 5
    terms[4][g1_idx[0][[top_row], [4 + block_i * 5, 4 + 5 * t + block_i]],
             g1_idx[1][[top_row], [4 + block_i * 5, 4 + 5 * t + block_i]]] = [[u, 1]]

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
    idx_block_1 = [idx[0][6:, :5], idx[1][6:, :5]]
    idx_block_2 = [idx[0][:5, 6:], idx[1][:5, 6:]]

    terms_vg_1 = create_top_left_factorisation_simple_vg(1 - val1, 1, shape, idx_block_1)
    terms_vg_2 = create_top_left_factorisation_simple_vg(1 - val2, 1, shape, idx_block_2)

    terms.extend(terms_vg_1)
    terms.extend(terms_vg_2)
    terms.append(np.zeros(shape))

    terms[-1][idx[0][5:7, 5:7], idx[1][5:7, 5:7]] = 1
    terms[-1][idx[0][0, 0], idx[1][0, 0]] = val1 * val2
    terms[-1][idx[0][0, 5], idx[1][0, 5]] = val1
    terms[-1][idx[0][5, 0], idx[1][5, 0]] = val2

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
            [[i], [t], [t + 1 + 5 * i + 1]],
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

    plt.figure()
    plt.imshow(N - np.sum(terms, axis=0))
    plt.gca().set_title("diff")

    print("expected_rank", expected_rank)
    print("len(terms)", len(terms))


if __name__ == "__main__":
    formula = "x * y + z - 0.32"
    solution = dict(
        x=0.4,
        y=0.8,
        z=1
    )
    test(formula, solution)

    plt.show()

    formula = "x * y + 2 * z + 2 *  z - 3 * u * v *  t - 4 * z - 5 * u * x + 1"
    solution = dict(
        x=0.5,
        y=0.6,
        z=0.5,
        v=0.5,
        u=0.4,
        t=0.5
    )
    test(formula, solution)