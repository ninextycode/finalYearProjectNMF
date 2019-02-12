import numpy as np
import torch


def get_random_lowrank_matrix(m, r, n):
    return np.random.rand(m, r) @ np.random.rand(r, n)


def get_time_ratio(errors_0, errors_1):
    # Rartio of times to reach certain cost function value
    max_log_error = min(np.max(np.log(errors_0[1:, 1])),
                        np.max(np.log(errors_1[1:, 1])))
    min_log_error = max(np.min(np.log(errors_0[:, 1])),
                        np.min(np.log(errors_1[:, 1])))

    n = 100
    error_space = np.linspace(min_log_error, max_log_error, n)
    time_rates = np.zeros(n)
    for err_i in range(n):
        time_0 = errors_0[np.log(errors_0[:, 1]) <= error_space[err_i], 0][0]
        time_1 = errors_1[np.log(errors_1[:, 1]) <= error_space[err_i], 0][0]
        time_rates[err_i] = time_0 / time_1

    return np.array([error_space, time_rates]).T


def compare_performance(V, inner_dim, time_limit,
                        W_init, H_init,
                        algo_dict_to_test,
                        kw_override={}):
    errors = {}
    for algo_name, algo in algo_dict_to_test.items():
        torch.cuda.empty_cache()
        kw_args_default = dict(V=V,
                               inner_dim=inner_dim,
                               record_errors=True,
                               time_limit=time_limit,
                               max_steps=np.inf,
                               epsilon=0,
                               W_init=W_init.copy(),
                               H_init=H_init.copy())

        kw_args = {**kw_args_default, **kw_override}
        _, _, errors[algo_name] = algo(**kw_args)
    return errors


def plot_performance_dict(errors, ax):
    keys = sorted(errors.keys())
    for name in keys:
        ls = "--" if "torch" in name else "-"
        ax.plot(errors[name][:, 0], np.log(errors[name][:, 1]), label=name, ls=ls)
    ax.legend()


def torch_algo_wrapper(algo, device="cuda"):
    def algo_w(*args, **kwargs):
        kwargs["V"] = torch.tensor(kwargs["V"], device=device)
        if "W_init" in kwargs.keys():
            kwargs["W_init"] = torch.tensor(kwargs["W_init"], device=device)
        if "H_init" in kwargs.keys():
            kwargs["H_init"] = torch.tensor(kwargs["H_init"], device=device)
        result = algo(*args, **kwargs)
        result = list(result)
        result[0] = result[0].to("cpu").numpy()
        result[1] = result[1].to("cpu").numpy()
        return result
    return algo_w


def errors_at_time_t_over_inner_dim(V, r_range, t, algo_dict):
    error_data = {algo_name: [] for algo_name in algo_dict.keys()}

    for r in r_range:
        W_init, H_init = nmf.mult.update_empty_initials(V, r, None, None)
        for algo_name, algo in algo_dict.items():
            W, H = algo(V=V, inner_dim=r,
                        record_errors=False,
                        time_limit=t,
                        max_steps=np.inf,
                        epsilon=0,
                        W_init=W_init.copy(),
                        H_init=H_init.copy())
            error = nmf.norms.norm_Frobenius(V - W @ H)
            error_data[algo_name].append([r, error])
    return {k: np.array(v) for k, v in error_data.items()}


def plot_errors_dict(dict_data, ax, log=False):
    for k, v in dict_data.items():
        ls = "--" if "torch" in k else "-"
        y_data = np.log(v[:, 1]) if log else v[:, 1]
        ax.plot(v[:, 0], y_data, label=k, ls=ls)
    ax.legend()
    return ax


def plot_ratios_gpu_algo(errors, ax, base="mult_torch"):
    ratios = get_time_ratio(errors[base], errors["nesterov_torch"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="nesterov_torch")

    ratios = get_time_ratio(errors[base], errors["mult_torch"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="mult_torch")

    ratios = get_time_ratio(errors[base], errors["pgrad_torch"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="pgrad_torch")

    ax.set_title("How faster is X than {} on GPU".format(base))
    ax.invert_xaxis()
    ax.legend()


def plot_ratios_cpu_algo(errors, ax, base="mult"):
    ratios = get_time_ratio(errors[base], errors["nesterov"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="nesterov")

    ratios = get_time_ratio(errors[base], errors["mult"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="mult")

    ratios = get_time_ratio(errors[base], errors["pgrad"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="pgrad")

    ax.set_title("How faster is X than {} on CPU".format(base))
    ax.invert_xaxis()
    ax.legend()


def plot_ratios_cpu_gpu(errors, ax):
    ratios = get_time_ratio(errors["nesterov"], errors["nesterov_torch"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="nesterov")

    ratios = get_time_ratio(errors["mult"], errors["mult_torch"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="mult")

    ratios = get_time_ratio(errors["pgrad"], errors["pgrad_torch"])
    ax.plot(ratios[:, 0], ratios[:, 1], label="pgrad")

    ax.set_title("How faster is X on GPU than on CPU")
    ax.invert_xaxis()
    ax.legend()