import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt

from diffopt.ridge import Ridge
from diffopt.logreg import LogReg
from diffopt.quadratic import Quadratic
from diffopt.utils import check_random_state


BENCH_NAME = "gradient_estimation"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def get_regression(n_samples=10, n_dim=50, n_features=100, reg=None,
                   random_state=None):
    if reg is None:
        reg = 1 / n_dim
    rng = check_random_state(random_state)
    D = rng.randn(n_dim, n_features)
    x = rng.randn(n_samples, n_dim)
    return x, D, reg


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


def run_benchmark(config):
    log_callbacks = ['z', 'g1', 'g2', 'g3']

    results = []
    for i, bench in enumerate(config):
        name = config[bench]['name']
        pb_func = config[bench]['pb_func']
        pb_args = config[bench]['pb_args']
        class_model = config[bench]['model']
        max_layer = config[bench]['max_layer']
        n_iters = np.unique(np.logspace(0, np.log10(max_layer), 50, dtype=int))
        print(f'\r{name} :', end='', flush=True)

        # Compute true minimizer
        model_star = class_model(n_layers=20 * max_layer, algorithm='gd')
        x, *loss_args = pb_func(**pb_args)
        g_star = model_star.get_grad_x(x, *loss_args, computation='analytic')
        z_star, _ = model_star.transform(x, *loss_args)

        model = class_model(n_layers=max(n_iters), algorithm='gd')
        _, log = model.transform(
            x, *loss_args, log_iters=n_iters, log_callbacks=log_callbacks,
            requires_grad=True)

        z_diff = np.array([np.linalg.norm((rec['z'] - z_star).ravel())
                           for rec in log])
        g1_diff = np.array([np.linalg.norm((rec['g1'] - g_star).ravel())
                            for rec in log])
        g2_diff = np.array([np.linalg.norm((rec['g2'] - g_star).ravel())
                            for rec in log])
        g3_diff = np.array([np.linalg.norm((rec['g3'] - g_star).ravel())
                            for rec in log])
        results.append(dict(bench=bench, g1_diff=g1_diff, g2_diff=g2_diff,
                            g3_diff=g3_diff, z_diff=z_diff, n_iters=n_iters))

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
    fig = plt.figure(figsize=(6.4 * n_plots, 7.2))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=n_plots,
                               height_ratios=[.05, .95])

    for i, bench in enumerate(config):
        name = config[bench]['name']
        b = df[df.bench == bench]

        ax = fig.add_subplot(gs[1, i])
        ax.set_title(name)

        n_iters = b.n_iters.iloc[0]
        g1_diff = b.g1_diff.iloc[0]
        g2_diff = b.g2_diff.iloc[0]
        g3_diff = b.g3_diff.iloc[0]
        z_diff = b.z_diff.iloc[0]
        ratio = g1_diff[0] / z_diff[0]

        handles = []
        handles.append(ax.semilogy(n_iters, g1_diff, linewidth=3)[0])
        handles.append(ax.semilogy(n_iters, g2_diff, linewidth=3)[0])
        handles.append(ax.semilogy(n_iters, g3_diff, linewidth=3)[0])
        handles.append(ax.semilogy(n_iters, z_diff * ratio,
                                   linewidth=3, color='k', linestyle='--')[0])
        plt.xlabel(r'Iterations $t$', fontsize=24)
        plt.ylabel('')
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.legend(handles, [r'$\|g_1^t - g^*\|$', r'$\|g_2^t - g^*\|$',
                        r'$\|g_3^t - g^*\|$', r'$\|z^t - z^*\|$'],
              loc='center', bbox_to_anchor=(0, .95, 1, .05), ncol=4,
              fontsize=36)
    plt.tight_layout()
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
        'quadratic': {
            'name': '(a) Quadratic',
            'pb_func': get_quadratic,
            'pb_args': dict(n_samples=10, n_dim=50, n_features=100,
                            random_state=42),
            'model': Quadratic,
            'max_layer': 400,
        },
        'ridge': {
            'name': '(b) Ridge Regression',
            'pb_func': get_regression,
            'pb_args': dict(n_samples=1, n_dim=50, n_features=100, reg=None,
                            random_state=29),
            'model': Ridge,
            'max_layer': 700,
        },
        'logreg_reg': {
            'name': '(c) Penalized Logistic Regression',
            'pb_func': get_regression,
            'pb_args': dict(n_samples=1, n_dim=50, n_features=100, reg=None,
                            random_state=9),
            'model': LogReg,
            'max_layer': 2000,
        },
        'logreg': {
            'name': '(d) Logistic Regression',
            'pb_func': get_regression,
            'pb_args': dict(n_samples=1, n_dim=5, n_features=9, reg=0,
                            random_state=8),
            'model': LogReg,
            'max_layer': 7000,
        },
    }

    if args.plot:
        plot_benchmark(config, file_name=args.file)
    else:
        run_benchmark(config)
