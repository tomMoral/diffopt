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


def time_gradient(gradient, model, args, n_layers, meta):
    if gradient == 'analytic':
        grad = model.gradient_beta_analytic
    elif gradient == 'autodiff':
        grad = model.gradient_beta
    t_start = time()
    grad(*args, output_layer=n_layers)
    delta_t = time() - t_start

    return dict(
        n_layers=n_layers, time=delta_t, gradient=gradient,
        **meta
    )


def run_benchmark(n_rep=50, max_layers=100, n_probe_layers=20,
                  gpu=False):
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
    gpu: boolean (default: False)
        If set to true, will also run the gradient computation on the GPU.
    """
    eps = 1
    dimensions = dict(n_alpha=1000, n_beta=500, point_dim=2)

    devices = ['cpu']
    if gpu:
        devices += ['cuda']

    layers = np.unique(np.logspace(0, np.log(max_layers), n_probe_layers,
                                   dtype=int))
    n_probe_layers = len(layers)

    layers = np.minimum(max_layers, layers)
    results = []
    for device in devices:
        for j in range(n_rep):
            alpha, beta, C, *_ = make_ot(**dimensions, random_state=None)
            args = check_tensor(alpha, beta, C, device=device)
            for i, nl in enumerate(layers):
                progress = (j*n_probe_layers + i) / (n_rep * n_probe_layers)
                print(f"\rBenchmark gradient computation on {device}: "
                      f"{progress:.1%}", end='', flush=True)
                for gradient in ['analytic', 'autodiff']:
                    model = Sinkhorn(
                        n_layers=nl, gradient_computation=gradient,
                        device=device, log_domain=False)
                    t_start = time()
                    model.gradient_beta(*args, eps=eps)
                    delta_t = time() - t_start
                    results.append(dict(
                        device=device, gradient=gradient, n_layers=nl,
                        time=delta_t, **dimensions
                    ))

    df = pd.DataFrame(results)
    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    tag = f"{tag}{'_gpu' if gpu else ''}"
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_{tag}.pkl"))


def plot_benchmark(file_name=None, gpu=False):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_*.pkl")
        file_list = glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)
    if gpu:
        df = df[df.device == 'cuda']
    else:
        df = df[df.device == 'cpu']

    fig, ax = plt.subplots()
    for gradient in ['analytic', 'autodiff']:
        curve = df[df.gradient == gradient].groupby('n_layers').time
        y = curve.median()
        ax.plot(y.index, y, label=gradient)
        ax.fill_between(y.index, y - curve.std(), y + curve.std(), alpha=.3)
    # curve = (2 * df[df.gradient == 'analytic'].groupby('n_layers').median())
    # curve.plot(y='time', ax=ax, label="2x analytic")
    # curve = (3 * df[df.gradient == 'analytic'].groupby('n_layers').median())
    # curve.plot(y='time', ax=ax, label="3x analytic")
    plt.legend(bbox_to_anchor=(-.02, 1.02, 1., .1), ncol=3,
               loc='center', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(y.index.min(), y.index.max())
    # plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_timing.pdf"))
    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', action='store_true',
                        help='If present, use GPU computations')
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
