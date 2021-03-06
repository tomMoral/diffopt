import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt

from diffopt.ridge import Ridge
from diffopt.p_norm import pNorm
from diffopt.logreg import LogReg
from diffopt.sinkhorn import Sinkhorn
# from diffopt.quadratic import Quadratic
from diffopt.utils import check_random_state
from diffopt.utils.viz import make_legend, STYLES
from diffopt.datasets.optimal_transport import make_ot


BENCH_NAME = "gradient_estimation"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def get_regularized_regression(n_samples=10, n_dim=50, n_features=100,
                               reg=None, random_state=None):
    if reg is None:
        reg = 1 / n_dim
    rng = check_random_state(random_state)
    D = rng.randn(n_dim, n_features)
    x = rng.randn(n_samples, n_dim)
    return x, D, reg


def get_regression(n_samples=10, n_dim=50, n_features=100, random_state=None):
    rng = check_random_state(random_state)
    U, _, V = np.linalg.svd(rng.randn(n_dim, n_features))
    D = U.dot(V[:n_dim])
    assert np.allclose(np.eye(n_dim), D.dot(D.T))
    x = rng.randn(n_samples, n_dim)
    return x, D


def get_quadratic(n_samples=100, n_dim=50, n_features=100, random_state=None):
    rng = check_random_state(random_state)
    v = rng.randn(n_features)
    u = rng.randn(n_dim)
    x = rng.randn(n_samples, n_dim)
    eig_A = np.linspace(.1, 1, n_dim)
    eig_B = np.linspace(.3, 1, n_features)

    m = n_samples + n_features + 100
    basis_A = np.linalg.svd(rng.randn(m, n_dim), full_matrices=False)
    basis_B = np.linalg.svd(rng.randn(m, n_features), full_matrices=False)
    A_ = np.dot(basis_A[0], eig_A[:, None] * basis_A[2])
    B_ = np.dot(basis_B[0], eig_B[:, None] * basis_B[2])

    A = np.dot(A_.T, A_)
    B = np.dot(B_.T, B_)
    C = np.dot(A_.T, B_) / 2
    return x, A, B, C, u, v


def get_optimal_transport_problem(n_alpha=100, n_beta=30, point_dim=2,
                                  n_samples=1, eps=1e-1, random_state=None):
    alphas, beta, C, *_ = make_ot(
        n_alpha=n_alpha, n_beta=n_beta, point_dim=point_dim,
        n_samples=n_samples, random_state=random_state)
    return alphas[:1], beta, C, eps


def run_benchmark(config):

    results = []
    for i, bench in enumerate(config):
        log_callbacks = config[bench].get('to_plot', None)
        if log_callbacks is None:
            log_callbacks = ['z', 'g1', 'g2', 'g3']
        name = config[bench]['name']
        pb_func = config[bench]['pb_func']
        pb_args = config[bench]['pb_args']
        class_model = config[bench]['model']
        model_args = config[bench]['model_args']
        max_layer = config[bench]['max_layer']
        n_iters = np.unique(np.logspace(0, np.log10(max_layer),
                            num=50, dtype=int))
        print(f'\r{name} :', end='', flush=True)

        # Compute true minimizer
        model_star = class_model(n_layers=20 * max_layer, **model_args)
        loss_args = pb_func(**pb_args)
        _, log_star = model_star.transform(
            *loss_args, log_iters=[model_star.n_layers],
            log_callbacks=['z', 'g1'])
        log_star = log_star[-1]
        assert not np.isnan(np.sum(log_star['z']))
        if bench == 'p_norm':
            log_star['g1'] = 0
        log_star['g2'] = log_star['g3'] = log_star['g1']

        model = class_model(n_layers=max(n_iters), **model_args)
        _, log = model.transform(
            *loss_args, log_iters=n_iters, log_callbacks=log_callbacks,
            requires_grad=True)

        diff = {k: np.array([np.linalg.norm((rec[k] - log_star[k]).ravel())
                             for rec in log]) for k in log_callbacks}
        results.append(dict(bench=bench, n_iters=n_iters, **diff))

    print("\rBench: done".ljust(40))
    df = pd.DataFrame(results)
    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_{tag}.pkl"))


def plot_benchmark(config, file_name=None):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_*.pkl")
        file_list = glob.glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)

    n_plots = len(config)
    fig = plt.figure(figsize=(6.4 * n_plots, 4.8))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=n_plots,
                               height_ratios=[.05, .95])

    for i, (bench, setting) in enumerate(config.items()):
        name = f"({chr(97 + i)}) {setting['name']}"
        to_plot = setting['to_plot']
        max_layer = setting['max_layer']
        b = df[df.bench == bench]

        ax = fig.add_subplot(gs[1, i])
        ax.set_title(name)

        xlim = (0, max_layer)

        for k in to_plot:
            ax.semilogy(
                b.n_iters.iloc[0], b[k].iloc[0], **STYLES[k])

        plt.xlabel(r'Iterations $t$', fontsize=24)
        plt.ylabel('')
        if i == 0:
            plt.ylabel(r'$|g^i_t - g^*|$')

        if bench == 'p_norm':
            # n_iters, g1, g2 = np.unique(b.n_iters.iloc[0]), b.g1, b.g2
            # n_iters = n_iters[-len(n_iters) // 2 + 2:]
            # ratio = g1.iloc[0][-1] * max(n_iters) ** 1.5
            # plt.plot(n_iters, ratio / n_iters ** 1.5, 'k--', alpha=.8,
            #          linewidth=3)

            # ratio = g2.iloc[0][-1] * max(n_iters) ** 3
            # plt.plot(n_iters, ratio / n_iters ** 3, 'k--', alpha=.8,
            #          linewidth=3)
            xscale = 'log'
            xlim = (1, xlim[1])
        else:
            xscale = 'linear'
        plt.xlim(xlim)
        ax.set_xscale(xscale)

    ax_legend = fig.add_subplot(gs[0, :])
    make_legend(ax_legend)

    plt.subplots_adjust(left=.05, bottom=.14, right=.98, top=.95,
                        wspace=.15, hspace=.33)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}.pdf"),
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run gradient computation for logistic regression')
    parser.add_argument('--plot', action='store_true',
                        help='Show the results from the benchmark')
    parser.add_argument('--file', type=str, default=None,
                        help='File to plot')
    args = parser.parse_args()

    config = {
        # 'quadratic': {
        #     'name': 'Quadratic',
        #     'model': Quadratic,
        #     'model_args': dict(algorithm='gd'),
        #     'max_layer': 400,
        #     'pb_func': get_quadratic,
        #     'pb_args': dict(n_samples=10, n_dim=50, n_features=100,
        #                     random_state=42),
        # },
        'ridge': {
            'name': 'Ridge Regression',
            'model': Ridge,
            'max_layer': 700,
            'model_args': dict(algorithm='gd'),
            'pb_func': get_regularized_regression,
            'pb_args': dict(n_samples=1, n_dim=50, n_features=100, reg=None,
                            random_state=29),
            'to_plot': ['z', 'g1', 'g2', 'g3'],
        },
        'logreg_reg': {
            'name': 'Regularized Logistic Regression',
            'model': LogReg,
            'max_layer': 2000,
            'model_args': dict(algorithm='gd'),
            'pb_func': get_regularized_regression,
            'pb_args': dict(n_samples=1, n_dim=50, n_features=100, reg=None,
                            random_state=9),
            'to_plot': ['z', 'g1', 'g2', 'g3'],
        },
        'p_norm': {
            'name': 'Least p-th norm',
            'model': pNorm,
            'max_layer': 500000,
            'model_args': dict(algorithm='gd', p=4),
            'pb_func': get_regression,
            'pb_args': dict(n_samples=1, n_dim=5, n_features=10,
                            random_state=8),
            'to_plot': ['z', 'g1', 'g2'],
        },
        'sinkhorn': {
            'name': 'Wasserstein Distance',
            'model': Sinkhorn,
            'max_layer': 60,
            'model_args': dict(log_domain=False),
            'pb_func': get_optimal_transport_problem,
            'pb_args': dict(n_alpha=100, n_beta=30, point_dim=2, n_samples=2,
                            random_state=53),
            'to_plot': ['g1', 'g2', 'g3', 'z'],
        },
    }

    make_ot()

    if args.plot:
        plot_benchmark(config, file_name=args.file)
    else:
        run_benchmark(config)
