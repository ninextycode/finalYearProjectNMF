from itertools import product
import numpy as np
import sympy as sym
import scipy.linalg
import matplotlib.pyplot as plt


variable_gadget_mat_1 = np.array([
    [0., 1., 1., 1., 1.],
    [1., 1., 1., 0., 0.],
    [0., 0., 1., 1., 0.],
    [0., 0., 0., 1., 1.],
    [0., 1., 0., 0., 1.]
])

def matrix_from_formula(formula):
    expanded_formula = sym.expand(formula)
    formula_list = expanded_expression_to_list(expanded_formula)
    positive_side = [t for t in formula_list if t[0] > 0]
    negative_side = [(-t[0], t[1]) for t in formula_list if t[0] < 0]

    l_max = np.sum([t[0] for t in positive_side]) + np.sum([t[0] for t in negative_side])

    mat_num_p, mat_var_p, ranges_p = matrices_for_poly_with_positive_coef(
        positive_side, "L", l_max, starting_index=0)
    mat_num_n, mat_var_n, ranges_n = matrices_for_poly_with_positive_coef(
        negative_side, "L", l_max, starting_index=len(positive_side))

    final_matrix_num = scipy.linalg.block_diag(*(mat_num_p + mat_num_n))
    final_matrix_var = scipy.linalg.block_diag(*(mat_var_p + mat_var_n))
    final_matrix_var[final_matrix_var == 0] = ""

    ranges = merge_dicts(ranges_p, ranges_n)
    r = len(positive_side) + len(negative_side)
    coef_for_term_products = np.sum([max(0, len(term[1]) - 1) for term in positive_side])\
                           + np.sum([max(0, len(term[1]) - 1) for term in negative_side])
    expected_rank = 5 * r + 9 * coef_for_term_products

    return final_matrix_num, final_matrix_var, ranges, int(expected_rank)


def matrices_for_poly_with_positive_coef(formula_list, var_result_name,
                                         sum_upper_bound, starting_index=0):
    matrices_num = []
    matrices_var = []
    ranges = {}

    poly_vars = []
    poly_coefs = []
    terms_wo_var = []

    for term in formula_list:
        matrices_num_l, matrices_var_l, ranges_l, last_v = \
            matrices_for_vars_prodct(term[1], starting_index)
        starting_index = starting_index + 1

        if last_v is None:
            terms_wo_var.append(term[0])
        else:
            poly_vars.append(last_v)
            poly_coefs.append(term[0])

        matrices_num.extend(matrices_num_l)
        matrices_var.extend(matrices_var_l)
        ranges = merge_dicts(ranges, ranges_l)

    terms_wo_var_sum = np.sum(terms_wo_var)
    if terms_wo_var_sum > 0:
        terms_wo_var = [terms_wo_var_sum]
    else:
        terms_wo_var = []

    all_coefs = poly_coefs + terms_wo_var

    poly_mat_num, poly_mat_var, poly_ranges = matrix_s(all_coefs, poly_vars,
                                                       var_result_name,
                                                       sum_upper_bound)

    matrices_num.append(poly_mat_num)
    matrices_var.append(poly_mat_var)
    ranges = merge_dicts(ranges, poly_ranges)

    return matrices_num, matrices_var, ranges


def matrices_for_vars_prodct(vars_list, idx):
    if len(vars_list) == 0:
        return [], [], {}, None
    if len(vars_list) == 1:
        return [], [], {vars_list[0]: [0, 1]}, vars_list[0]

    expanded_vars = expanded_variables_names(vars_list, "v{}".format(idx))

    matrices_num = []
    matrices_var = []
    ranges = {}

    last_v = None
    for v, u1, u2 in expanded_vars:
        num, var, local_ranges = matrix_p([u1, u2], v)
        last_v = v
        matrices_num.append(num)
        matrices_var.append(var)
        ranges = merge_dicts(ranges, local_ranges)

    return matrices_num, matrices_var, ranges, last_v


def merge_dicts(d1, d2):
    return {**d1, **d2}


def expanded_variables_names(variables, prefix):
    us = [["_{}1".format(prefix), variables[0], variables[1]]]\
         + [["_{}{}".format(prefix, j), "_{}{}".format(prefix, j - 1), variables[j]]
            for j in range(2, len(variables))]
    return us


def expanded_expression_to_list(formula):
    if formula.is_symbol:
        return [(1.0, [str(formula)])]
    if formula.is_real:
        return [(float(formula), [])]

    terms = formula.args
    formula_list = []
    for term in terms:
        formula_list.append(term_to_tuple(term))
    return formula_list


def term_to_tuple(term):
    if term.is_real:
        return float(term), []
    if term.is_symbol:
        return 1.0, str(term)

    term_args = term.args
    coef = 1
    vars = []
    for part in term_args:
        if part.is_real:
            coef = float(part)
        if part.is_Pow:
            for _ in range(part.exp):
                vars.append(str(part.base))
        if part.is_symbol:
            vars.append(str(part))

    return coef, vars


def remove_variables(num, var, ranges, expected_rank):
    if num.shape != var.shape:
        raise Exception("Shpes do not match")
    vars_in_matrix = set(var.ravel()) - {""}
    if set(ranges.keys()) != vars_in_matrix:
        raise Exception("Given ranges and variable matrix contain different variable ranges"
                        "{} and {}".format(set(ranges.keys()), (set(var.ravel()) - {""})))

    expected_rank_expanded = expected_rank + 4 * len(vars_in_matrix) \
                             + 5 * np.sum([np.sum(var == v) for v in vars_in_matrix])

    for v in ranges.keys():
        print("remove_variable", v)
        num, var = remove_variable_no_rearrangement(num, var, ranges, v)
    return num, var, int(expected_rank_expanded)


def remove_variable_no_rearrangement(num, var, ranges, v):
    num = num.astype(float)
    min_v, max_v = ranges[v]
    if min_v == max_v:
        num[var == v] = min_v
        var[var == v] = ""
        return num, var

    idx_g, idx_g1, t = indices_in_G1_shape(num, var, v)

    N = 1; Q = 1; M = max_v + 1
    P = 1 / ((max_v - min_v) + 1)

    num = np.insert(num, num.shape[0], values=[[0] for _ in range(5 * t + 5)], axis=0)
    num = np.insert(num, num.shape[1], values=[[0] for _ in range(5 * t + 4)], axis=1)

    var = np.insert(var, var.shape[0], values=[[""] for _ in range(5 * t + 5)], axis=0)
    var = np.insert(var, var.shape[1], values=[[""] for _ in range(5 * t + 4)], axis=1)

    top_block_idx = [slice(5), slice(4)]
    num[idx_g[0][top_block_idx],  idx_g[1][top_block_idx]] = [
        [N, N, 0, 0],
        [0, N, N, 0],
        [0, 0, N, N],
        [N, 0, 0, N],
        [N, N, N, N]
    ]

    block_for_N_idx = [[3, 4], slice(4, (4 + t))]
    num[idx_g[0][block_for_N_idx], idx_g[1][block_for_N_idx]] = N

    block_for_1s_idx = [4, slice(4 + t, 4 + 2 * t)]
    num[idx_g[0][block_for_1s_idx], idx_g[1][block_for_1s_idx]] = 1

    block_for_Qs_idx = [slice(5, 5 + t), slice(4, 4 + t)]
    diag_idx = np.diag_indices(t)
    num[idx_g[0][block_for_Qs_idx][diag_idx], idx_g[1][block_for_Qs_idx][diag_idx]] = Q

    identity_1_idx = [slice(5, 5 + t), slice(4 + t, 4 + 2 * t)]
    num[idx_g[0][identity_1_idx][diag_idx], idx_g[1][identity_1_idx][diag_idx]] = 1

    identity_2_idx = [slice(5 + t, 5 + 2 * t), slice(4, 4 + t)]
    num[idx_g[0][identity_2_idx][diag_idx], idx_g[1][identity_2_idx][diag_idx]] = 1

    block_for_Ms_idx = [slice(5 + t, 5 + 2 * t), slice(4 + t, 4 + 2 * t)]
    num[idx_g[0][block_for_Ms_idx][diag_idx], idx_g[1][block_for_Ms_idx][diag_idx]] = M

    for i in range(t):
        gadget_idx = [slice(5 + i * 5, 5 + i * 5 + 5), slice(4 + i * 5, 4 + i * 5 + 5)]
        num[idx_g1[0][gadget_idx], idx_g1[1][gadget_idx]] += (Q - P) * variable_gadget_mat_1

    return num, var

def indices_in_G1_shape(num, var, v):
    rows_idx_v, cols_idx_v = np.where(var == v)

    if len(rows_idx_v) == 0:
        raise Exception("Variable \"{}\" not found".format(v))

    if len(rows_idx_v) != len(set(rows_idx_v)) or \
            len(cols_idx_v) != len(set(cols_idx_v)):
        raise Exception("Multiple variables \"{}\" in one row or column".format(v))

    t = len(rows_idx_v)

    offset_rows = 5 * t + 5
    offset_cols = 5 * t + 4

    rows_idx_g = [i + num.shape[0] for i in range(5)] +\
                 [5 * i + 5 + num.shape[0] for i in range(t)] + \
                 list(rows_idx_v)
    cols_idx_g = [i + num.shape[1] for i in range(4)] +\
                 [5 * i + 4 + num.shape[1] for i in range(t)] +\
                 list(cols_idx_v)

    rows_idx_g1 = [i + num.shape[0] for i in range(offset_rows)]
    cols_idx_g1 = [i + num.shape[1] for i in range(offset_cols)]

    return np.meshgrid(rows_idx_g, cols_idx_g, indexing='ij'), \
           np.meshgrid(rows_idx_g1, cols_idx_g1, indexing='ij'), t


def remove_variable(num, var, ranges, v):
    num = num.astype(float)
    min_v, max_v = ranges[v]
    if min_v == max_v:
        num[var == v] = min_v
        var[var == v] = ""
        return num, var

    num, var, top_left, t = form_diagonal(num, var, v)

    N = 1
    Q = 1
    P = 1 / ((max_v - min_v) + 1)
    M = max_v + 1

    num, var = fill_G1(num, var, N, Q, M, top_left, t)
    num, var = fill_G(num, var, Q, P, top_left, t)

    return num, var


def form_diagonal(num, var, v):
    rows_idx, cols_idx = np.where(var == v)

    if len(rows_idx) == 0:
        raise Exception("Variable \"{}\" not found".format(v))

    if len(rows_idx) != len(set(rows_idx)) or \
            len(cols_idx) != len(set(cols_idx)):
        raise Exception("Multiple variables \"{}\" in one row or column".format(v))

    idx_sorted_by_cols = np.argsort(cols_idx)
    rows_i_sorted_by_cols = rows_idx[idx_sorted_by_cols]
    sorted_cols_idx = cols_idx[idx_sorted_by_cols]

    first_row = np.min(rows_idx)
    first_col = np.min(cols_idx)

    new_row_idx = list(range(first_row, first_row + rows_idx.shape[0]))
    new_col_idx = list(range(first_col, first_col + cols_idx.shape[0]))

    num, var = swap_rows(rows_i_sorted_by_cols, new_row_idx, num, var)
    num, var = swap_cols(sorted_cols_idx, new_col_idx, num, var)

    return num, var, (first_row, first_col), rows_idx.shape[0]


def swap_rows(idx_old, idx_new, *Mats):
    idx_old, idx_new = swap_extend_idx(idx_old, idx_new)
    expanded_idx_old = [idx_old, slice(None)]
    expanded_idx_new = [idx_new, slice(None)]
    return swap_subroutine(expanded_idx_old, expanded_idx_new, *Mats)


def swap_cols(idx_old, idx_new, *Mats):
    idx_old, idx_new = swap_extend_idx(idx_old, idx_new)
    expanded_idx_old = [slice(None), idx_old]
    expanded_idx_new = [slice(None), idx_new]
    return swap_subroutine(expanded_idx_old, expanded_idx_new, *Mats)


def swap_extend_idx(idx_old, idx_new):
    idx_all_set = set(idx_old).union(idx_new)

    add_to_idx_old = list(idx_all_set - set(idx_old))
    add_to_idx_new = list(idx_all_set - set(idx_new))

    idx_old_ext = list(idx_old) + add_to_idx_old
    idx_new_ext = list(idx_new) + add_to_idx_new

    return idx_old_ext, idx_new_ext


def swap_subroutine(idx_old, idx_new, *Mats):
    for i in range(len(Mats)):
        Mats[i][idx_new] = Mats[i][idx_old]

    if len(Mats) == 0:
        return None
    if len(Mats) == 1:
        return Mats[0]
    return Mats


def fill_G1(num, var, N, Q, M, top_left, t):
    num = np.insert(num, top_left[0], values=[[0] for _ in range(t + 5)], axis=0)
    num = np.insert(num, top_left[1], values=[[0] for _ in range(t + 4)], axis=1)

    var = np.insert(var, top_left[0], values=[[""] for _ in range(t + 5)], axis=0)
    var = np.insert(var, top_left[1], values=[[""] for _ in range(t + 4)], axis=1)

    new_area = [slice(top_left[0], (top_left[0] + 5 + 2 * t)),
                  slice(top_left[1], (top_left[1] + 4 + 2 * t))]

    num[new_area][:5, :4] = [
        [N, N, 0, 0],
        [0, N, N, 0],
        [0, 0, N, N],
        [N, 0, 0, N],
        [N, N, N, N]
    ]

    num[new_area][[3, 4], 4:(4 + t)] = N
    num[new_area][4,(4 + t):(4 + 2 * t)] = 1

    num[new_area][5:(5 + t), 4:(4 + t)][np.diag_indices(t)] = Q

    num[new_area][5: (5 + t), (4 + t):(4 + 2 * t)][np.diag_indices(t)] = 1
    num[new_area][(5 + t):(5 + 2 * t), 4:(4 + t)][np.diag_indices(t)] = 1

    num[new_area][(5 + t):(5 + 2 * t), (4 + t):(4 + 2 * t)][np.diag_indices(t)] = M

    return num, var


def fill_G(num, var, Q, P, top_left, t):
    for i in reversed(range(t)):
        position_idx = (top_left[0] + 5 + i, top_left[1] + 4 + i)
        num, var = apply_variable_gadget_to_single(num, var, position_idx, Q - P)
    return num, var


def matrix_s(s_list, var_coef_names, var_result_name, sum_upper_bound=None):
    if sum_upper_bound is None:
        sum_upper_bound = np.sum(s_list)
    l = len(s_list)
    num = np.full((2 * l + 1, l + 1),  0.0)
    num[:, 1:][np.diag_indices(l)] = 1
    num[:len(var_coef_names), 0] = -1
    num[l, 0] = -1
    num[len(var_coef_names):l, 0] = 1  # constants
    num[(l+1):, 1:][np.diag_indices(l)] = 1
    num[(l + 1):, 0] = 1
    num[l, 1:] = s_list

    var = np.full(num.shape, "", dtype=object)
    var[:len(var_coef_names), 0] = var_coef_names
    var[l, 0] = var_result_name

    ranges = {var: [0.0, 1.0] for var in var_coef_names}
    ranges[var_result_name] = [0, sum_upper_bound]

    for i in reversed(range(l)):
        num, var = apply_variable_gadget_to_single(num, var, (l + 1 + i, 0), 1)

    return num, var, ranges


def matrix_p(var_names, var_product_name):
    num = np.full((3, 3), 1.0)

    var = np.full((3, 3), "", dtype=object)
    var[0, 0] = var_product_name
    var[0, 1] = var_names[0]
    var[1, 0] = var_names[1]

    num[0, 1] = -1
    num[1, 0] = -1
    num[0, 0] = -1

    ranges = {var: [0.0, 1.0] for var in var_names}
    ranges[var_product_name] = [0.0, 1.0]

    num, var = apply_variable_gadget_to_single(num, var, (0, 2), 1)
    num, var = apply_variable_gadget_to_single(num, var, (2 + 4, 0), 1)
    return num, var, ranges


def apply_variable_gadget_to_single(num, var, top_left, M):
    num = np.insert(num, top_left[0]+1, values=[[0] for _ in range(4)], axis=0)
    num = np.insert(num, top_left[1]+1, values=[[0] for _ in range(4)], axis=1)

    var = np.insert(var, top_left[0]+1, values=[[""] for _ in range(4)], axis=0)
    var = np.insert(var, top_left[1]+1, values=[[""] for _ in range(4)], axis=1)

    new_area = [slice(top_left[0], (top_left[0] + 5)),
                  slice(top_left[1], (top_left[1] + 5))]
    num[new_area][0, 1:5] = M
    num[new_area][1, 0] = M

    num[new_area][1:5, 1:5] = [
        [M, M, 0, 0],
        [0, M, M, 0],
        [0, 0, M, M],
        [M, 0, 0, M]
    ]

    return num, var



from multiprocessing import Pool
from nmf.pgrad import factorise_Fnorm_subproblems
from nmf.nesterov import factorise_Fnorm


def test_trasform_G():


    np.random.seed(0)
    N = np.array([i / 15 for i in range(35)], dtype=float).reshape((7, 5))
    N[2, 1] = -1
    N[1, 2] = -1
    N[4, 3] = -1

    V = np.full(N.shape, "", dtype=object)
    V[N < 0] = "x"

    plot(N, V); plt.title("N original")

    N_, V_ = remove_variable(N.copy(), V.copy(), {"x": [0, 1]}, "x")

    plot(N_, V_); plt.title("N x removed")

    N_, V_ = remove_variable_no_rearrangement(N.copy(), V.copy(), {"x": [0, 1]}, "x")

    plot(N_, V_); plt.title("N x removed alternative")

def get_errors(N, W_init, H_init, q_):
    W, H, errors_i = factorise_Fnorm(N, q_, W_init=W_init, H_init=H_init,
                                     n_steps=200, epsilon=0,
                                     record_errors=True)
    return np.log(errors_i[:, 0])


def test2(formula):
    mat_num, mat_var, ranges, expected_rank = matrix_from_formula(formula)
    plot(mat_num, mat_var)

    N, V, q_ = remove_variables(mat_num, mat_var, ranges, expected_rank)
    print("ranges", ranges)
    print("expected_rank", expected_rank)
    print("expected_rank_expanded", q_)

    plot(N, V)

    with Pool() as p:
        map_res = p.starmap_async(get_errors,
                           [
                               (N.copy(),
                                1 - np.random.rand(N.shape[0], q_),
                                1 - np.random.rand(q_, N.shape[1]),
                                q_)
                            for _ in range(1000)])
        W, H, errors_i = factorise_Fnorm_subproblems(N.copy(), q_, n_steps=200, epsilon=0,
                                                     record_errors=True)
        errors_i = np.log(errors_i[:, 0])
        errors = map_res.get()

    errors = np.column_stack(errors)
    min_err = np.min(errors, axis=1)

    plt.figure()
    plt.plot(min_err, color="tab:red", linewidth=5)
    plt.plot(errors, color="tab:blue")
    plt.plot(errors_i, color="tab:orange")
    plt.title(formula)

    plt.figure()
    plt.subplot(121)
    plt.imshow(N)
    plt.title("N")
    plt.subplot(122)
    plt.imshow(W @ H)
    plt.title("W @ H")

    plt.show()

    return mat_num, mat_var, N, W, H, q_


def plot(num, var):
    plt.figure()
    plt.imshow(num)
    for i, j in product(range(num.shape[0]), range(num.shape[1])):
        plt.text(j, i, var[i, j], color="white",
                 backgroundcolor="black",
                 horizontalalignment="center")



from main import plot_factorisation

def text3(formula):
    mat_num, mat_var, ranges, expected_rank = matrix_from_formula(formula)
    plot(mat_num, mat_var)

    N, V, q_ = remove_variables(mat_num, mat_var, ranges, expected_rank)
    print("ranges", ranges)
    print("expected_rank", expected_rank)
    print("expected_rank_expanded", q_)

    plot(N, V)


    W, H, errors_i = factorise_Fnorm(N, q_, n_steps=100, epsilon=0, record_errors=True)


    plt.figure()
    plt.plot(np.log(errors_i[:, 0]), color="tab:red", linewidth=5)
    plt.title(formula)

    plt.figure()
    plt.subplot(121)
    plt.imshow(N)
    plt.title("N")
    plt.subplot(122)
    plt.imshow(W @ H)
    plt.title("W @ H")

    plot_factorisation(W, H, (6, 6))
    plt.show()
    return mat_num, mat_var, N, W, H, q_


text3("2 * x - 1")

plt.show()