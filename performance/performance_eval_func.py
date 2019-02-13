import numpy as np
import torch
import nmf.mult
from scipy.interpolate import interp1d


colors_default = {
    "mult": 'tab:blue',
    "pgrad": "tab:green",
    "nesterov": "tab:red"
}


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

    time_by_error_0 = interp1d(np.log(errors_0[:, 1]), errors_0[:, 0])
    time_by_error_1 = interp1d(np.log(errors_1[:, 1]), errors_1[:, 0])

    time_rates = time_by_error_0(error_space) / time_by_error_1(error_space)
    return np.array([error_space, time_rates]).T


def compare_performance(V, inner_dim, time_limit,
                        W_init, H_init,
                        algo_dict_to_test,
                        kw_override={}):
    errors = {}
    for algo_name, algo in algo_dict_to_test.items():
        print("Starting " + algo_name)
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


def plot_performance_dict(errors, ax, colors=colors_default):
    keys = sorted(errors.keys())
    for name in keys:
        ls = "--" if "torch" in name else "-"
        kwargs = dict(label=name, ls=ls)
        for color_name, color in colors.items():
            if color_name == name.split("_")[0]:
                kwargs["color"] = color
        ax.plot(errors[name][:, 0], np.log(errors[name][:, 1]), **kwargs)
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


def plot_errors_dict(dict_data, ax, log=False, title=None, x_lbl=None):
    for k, v in dict_data.items():
        ls = "--" if "torch" in k else "-"
        y_data = np.log(v[:, 1]) if log else v[:, 1]
        ax.plot(v[:, 0], y_data, label=k, ls=ls)
    ax.set_title("Objective function")

    y_lbl = "log(error)" if log else "error"
    ax.set_ylabel(y_lbl)
    if x_lbl is not None:
        ax.set_xlabel(x_lbl)

    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_ratios(errors, ax, base, selected_algs=None, colors=colors_default):
    if selected_algs is None:
        selected_algs = errors.keys()
    for algo_name in selected_algs:
        kwargs = dict(label=algo_name)
        algo_name_perfix = algo_name.split("_")[0]
        if algo_name_perfix in colors.keys():
            kwargs["color"] = colors_default[algo_name_perfix]
        ratios = get_time_ratio(errors[base], errors[algo_name])
        ax.plot(ratios[:, 0], ratios[:, 1], **kwargs)
    ax.set_xlabel("log(error)")
    ax.set_ylabel("time ratio")
    ax.invert_xaxis()
    ax.legend()


def plot_ratios_gpu_algo(errors_dict, axes, selected_algs=None, colors=colors_default):
    if selected_algs is None:
        selected_algs = [n for n in errors_dict.keys() if len(n.split("_")) > 0 and n.split("_")[-1] == "torch"]

    key = lambda n: errors_dict[n][-1, 1]
    names_by_error = sorted(selected_algs, key=key, reverse=True)

    for i, base in zip(range(len(names_by_error) - 1), names_by_error):
        plot_ratios(errors_dict, axes[i], base=base, selected_algs=names_by_error[i:], colors=colors)
        axes[i].set_title("How faster is X than {} on GPU".format(base))


def plot_ratios_cpu_algo(errors_dict, axes, selected_algs=None, colors=colors_default):
    if selected_algs is None:
        selected_algs = [n for n in errors_dict.keys() if len(n.split("_")) == 0 or n.split("_")[-1] != "torch"]

    key = lambda n: errors_dict[n][-1, 1]
    names_by_error = sorted(selected_algs, key=key, reverse=True)

    for i, base in zip(range(len(names_by_error) - 1), names_by_error):
        plot_ratios(errors_dict, axes[i], base=base, selected_algs=names_by_error[i:], colors=colors)
        axes[i].set_title("How faster is X than {} on CPU".format(base))


def plot_ratios_cpu_gpu(errors_dict, ax, colors=colors_default):
    for name in errors_dict.keys():
        if len(name.split("_")) > 0 and name.split("_")[-1] == "torch":
            continue

        kwargs = dict(label=name)
        if name in colors.keys():
            kwargs["color"] = colors_default[name]

        ratios = get_time_ratio(errors_dict[name], errors_dict[name + "_torch"])
        ax.plot(ratios[:, 0], ratios[:, 1], **kwargs)

    ax.set_title("How faster is X on GPU than on CPU")
    ax.invert_xaxis()
    ax.legend()