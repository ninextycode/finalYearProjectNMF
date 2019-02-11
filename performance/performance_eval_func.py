import numpy as np
import torch


def get_random_lowrank_matrix(m, r, n):
    return np.random.rand(m, r) @ np.random.rand(r, n)


def get_time_ratio(errors_0, errors_1):
    # Rartio of times to reach certain cost function value
    max_log_error = max(np.max(np.log(errors_0[1:, 1])),
                        np.max(np.log(errors_1[1:, 1])))
    min_log_error = min(np.min(np.log(errors_0[:, 1])),
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


def plot_performance(errors, ax):
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
