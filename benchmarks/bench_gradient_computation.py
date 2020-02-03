import os
import numpy as np
import pandas as pd
from glob import glob
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

from diffopt.sinkhorn import Sinkhorn
from diffopt.utils import check_tensor
from diffopt.datasets.optimal_transport import make_ot


BENCH_NAME = "gradient_computations"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def run_benchmark(n_rep=50, max_layers=100, n_probe_layers=20,
                  gpu=None):
    """Benchmark for the gradient computation time (analytic vs autodiff)

    Parameters:
    -----------
    n_rep: int (default: 50)
        Number of repetition for the benchmark. For each repetition, a new
        problem is created and the gradient are computed for different number
        of layers.
    max_layers: int (default: 100)
        Maximal number of layers. The benchmark is run for different number of
        n_layers which are chosen in log-scale between 1 and max_layers.
    n_probe_layers: int (default: 20)
        Number of number of layers chosen in the log-scale.
    gpu: int (default: none)
        If not None, use GPU number `gpu` to run the gradient computation.
    """
    eps = 1
    dimensions = dict(n_alpha=1000, n_beta=500, point_dim=2, n_samples=100)

    device = f'cuda:{gpu}' if gpu is not None else None

    layers = np.unique(np.logspace(0, np.log(max_layers), n_probe_layers,
                                   dtype=int))
    n_probe_layers = len(layers)

    layers = np.minimum(max_layers, layers)
    results = []
    for j in range(n_rep):
        alpha, beta, C, *_ = make_ot(**dimensions, random_state=None)
        args = check_tensor(alpha, beta, C, device=device)
        for i, nl in enumerate(layers):
            progress = (j*n_probe_layers + i) / (n_rep * n_probe_layers)
            print(f"\rBenchmark gradient computation on {device}: "
                  f"{progress:.1%}", end='', flush=True)
            for gradient in ['analytic', 'autodiff', 'implicit']:
                model = Sinkhorn(
                    n_layers=nl, gradient_computation=gradient,
                    device=device, log_domain=False)
                t_start = time()
                model.gradient_beta(*args, eps=eps)
                delta_t = time() - t_start
                results.append(dict(
                    gradient=gradient, n_layers=nl, time=delta_t, **dimensions
                ))

    df = pd.DataFrame(results)
    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_{tag}.pkl"))


def plot_benchmark(file_name=None, gpu=False):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_*.pkl")
        file_list = glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)

    fig, ax = plt.subplots()
    for i, gradient in enumerate(['analytic', 'autodiff', 'implicit']):
        curve = df[df.gradient == gradient].groupby('n_layers').time
        y = curve.median()
        ax.plot(y.index, y, label=f"$T(g_{i+1})$")
        ax.fill_between(y.index, y - curve.quantile(.25),
                        y + curve.quantile(.75), alpha=.3)
    curve = (3 * df[df.gradient == 'analytic'].groupby('n_layers').median())
    curve.plot(y='time', ax=ax, label="$3T(g_1)$", color='C0', linestyle='--')
    # curve = (3 * df[df.gradient == 'analytic'].groupby('n_layers').median())
    # curve.plot(y='time', ax=ax, label="3x analytic")
    plt.legend(bbox_to_anchor=(-.02, 1.05, 1., .1), ncol=3,
               loc='lower center', fontsize=18)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Runtime [sec]')
    ax.set_xlabel('Iteration $t$')
    ax.set_xlim(y.index.min() + 1, y.index.max())
    # plt.subplots_adjust(.05, .05, .95, .95)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_timing.pdf"),
                bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', type=int, default=None,
                        help='If present, use GPU # for computations')
    parser.add_argument('--plot', action='store_true',
                        help='Show the results from the benchmark')
    parser.add_argument('--file', type=str, default=None,
                        help='File to plot')
    parser.add_argument('--max-layers', type=int, default=100,
                        help='Maximal number of layers to test')
    parser.add_argument('--n-rep', type=int, default=100,
                        help='Number of repetition for each timing')
    parser.add_argument('--n-probe-layers', '-n', type=int, default=20,
                        help='Number of points in the curve. The points are '
                        'logspaced between 1 and `max_layers`')
    args = parser.parse_args()

    if args.plot:
        plot_benchmark(file_name=args.file, gpu=args.gpu)
    else:
        run_benchmark(n_rep=args.n_rep, max_layers=args.max_layers,
                      n_probe_layers=args.n_probe_layers, gpu=args.gpu)
