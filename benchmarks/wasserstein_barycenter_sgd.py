import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from time import time
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt

from diffopt.sinkhorn import Sinkhorn
from diffopt.utils import check_tensor
from diffopt.datasets.optimal_transport import make_ot

BENCH_NAME = "wasserstein_barycenter_sgd"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


N_INNER_FULL = 500


def wasserstein_barycenter_sgd(alphas, C, eps, n_epochs, n_inner, gradient,
                               step_size=.1, device=None):
    n_samples, n_alpha = alphas.shape
    # Generate the initial barycenter as the uniform distribution
    beta = np.ones(n_alpha) / n_alpha

    alphas, beta, C = check_tensor(alphas, beta, C, device=device)

    sinkhorn = Sinkhorn(n_layers=n_inner, log_domain=False,
                        gradient_computation=gradient, device=device)

    sinkhorn_full = Sinkhorn(
        n_layers=N_INNER_FULL, log_domain=False,
        gradient_computation='analytic', device=device)

    results = []
    it_loss = np.logspace(0, np.log10(n_epochs*n_samples),
                          num=n_epochs*n_samples//10, dtype=int)
    t_start = time()
    for id_epoch in range(n_epochs):
        print(f"{id_epoch/n_epochs:.1%}".rjust(7, '.') + '\b' * 7,
              end='', flush=True)
        for i in range(n_samples):
            it = id_epoch * n_samples + i
            G = sinkhorn._gradient_beta(alphas[i], beta, C, eps)
            with torch.no_grad():
                beta *= torch.exp(-step_size * G)
                beta /= beta.sum()

            if it in it_loss:
                delta_t = time() - t_start
                with torch.no_grad():
                    G_star, loss = sinkhorn_full.gradient_beta(
                        alphas, beta, C, eps, return_loss=True)
                assert not np.isnan(loss)
                results.append(dict(
                    n_inner=n_inner, gradient=gradient, step_size=step_size,
                    iteration=it, time=delta_t, loss=loss,
                    norm_gstar=np.linalg.norm(G_star.ravel()),
                    g_diff=np.linalg.norm(G_star.ravel() -
                                          G.cpu().numpy().ravel()),
                    best=n_inner == N_INNER_FULL
                ))
                t_start = time()

    print("done".rjust(7, '.'))
    return beta, results


def run_benchmark(n_samples=10, n_alpha=100, eps=1, n_epochs=300,
                  step_size=.1, max_layers=64, gpu=None):
    """Benchmark for the wasserstein barycenter computation time

    Parameters:
    -----------
    n_samples: int (default: 10)
        Number of distribution to compute the barycenter from.
    n_alpha: int (default: 100)
        Number of point in the support of the distributions.
    eps: float (default: 1)
        Entropy regularization parameter for the Wasserstein distance.
    n_epochs: int (default: 300)
        Maximal number of iteration run for the gradient descent algorithm.
    step_size: float (default: .1)
        Step size for the gradient descent.
    max_layers: int (default: 64)
        The benchmark is computed for a number of inner layers from 1 to
        max_layers in logscale. The max_layer will be rounded to the largest
        power of 2 below.
    gpu: int (default: None)
        If set, will run the computation on GPU number `gpu`.
    """

    device = f'cuda:{gpu}' if gpu is not None else None

    alphas, _, C, *_ = make_ot(n_alpha=n_alpha, n_beta=n_alpha, point_dim=2,
                               n_samples=n_samples)

    results = []
    max_layers = int(np.log2(max_layers))
    for n_inner in np.logspace(0, max_layers, num=max_layers + 1,
                               base=2, dtype=int):
        for gradient in ['autodiff', 'analytic']:
            print(f"Fitting {gradient}[{n_inner}]:", end='', flush=True)
            beta_star, res = wasserstein_barycenter_sgd(
                alphas, C, eps, n_epochs=n_epochs, n_inner=n_inner,
                step_size=step_size, gradient=gradient, device=device)
            results.extend(res)

    print("Fitting optimal barycenter:", end='', flush=True)
    beta_star, res = wasserstein_barycenter_sgd(
        alphas, C, eps=eps, n_epochs=n_epochs + 100, n_inner=N_INNER_FULL,
        step_size=step_size, gradient='analytic', device=device)
    results.extend(res)

    df = pd.DataFrame(results)
    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_{tag}.pkl"))


def plot_benchmark(file_name=None):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_*.pkl")
        file_list = glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)

    df_best = df[df.best]
    min_loss = df_best.loss.min() - 1e-6
    df = df[~df.best]

    list_n_layers = np.unique(df.n_inner)
    n_col = len(list_n_layers)

    # Create a grid of
    fig = plt.figure(figsize=(12.8, 4.8))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=3,
                               height_ratios=[.05, .95],
                               width_ratios=[.49, .49, .02])
    axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

    styles = {'analytic': '--', 'autodiff': '-'}
    cmap = plt.cm.get_cmap('viridis', n_col)
    colors = {k: cmap(i) for i, k in enumerate(list_n_layers)}

    axes[0].loglog(df_best.iteration, df_best.loss - min_loss,
                   'k')
    axes[1].loglog(np.cumsum(df_best.time), df_best.loss - min_loss, 'k')

    ls_legend_handle, ls_legend_label = [], []
    for gradient in np.unique(df.gradient):
        df_grad = df[df.gradient == gradient]
        style = styles[gradient]
        ls_legend_handle.append(
            mpl.lines.Line2D([0], [0], color='k', linestyle=style, lw=2))
        ls_legend_label.append(gradient)
        for n_layers in np.unique(df_grad.n_inner):
            color = colors[n_layers]
            to_plot = df_grad[df_grad.n_inner == n_layers]

            axes[0].loglog(
                to_plot.iteration, to_plot.loss - min_loss,
                label=gradient, color=color, linestyle=style,
                linewidth=2)
            axes[1].loglog(
                np.cumsum(to_plot.time), to_plot.loss - min_loss,
                label=gradient, color=color, linestyle=style,
                linewidth=2)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.legend(ls_legend_handle, ls_legend_label, ncol=2, loc='center')

    # Boundaries are selected between the power of 2 in logscale
    boundaries = 2 ** (np.arange(-1, 2 * n_col) / 2)[::2]
    mpl.colorbar.ColorbarBase(
            cmap=plt.get_cmap('viridis', n_col),
            norm=mpl.colors.BoundaryNorm(boundaries, n_col),
            ticks=list_n_layers, format='%d', ax=fig.add_subplot(gs[1, 2])
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_loss.pdf"))
    plt.show()

    # import IPython; IPython.embed(colors='neutral')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', type=int, default=None,
                        help='If present, use GPU computations')
    parser.add_argument('--plot', action='store_true',
                        help='Show the results from the benchmark')
    parser.add_argument('--file', type=str, default=None,
                        help='File to plot')
    parser.add_argument('--max-layers', type=int, default=64,
                        help='Maximal number of layers to test')
    parser.add_argument('--n-alpha', type=int, default=1000,
                        help='Size of the support of the distributions.')
    parser.add_argument('--n-epochs', type=int, default=2000,
                        help='Maximal number of iteration for the gradient '
                        'descent')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Number of distribution to average.')
    parser.add_argument('--step-size', type=float, default=.1,
                        help='Step size for the mirror descent algorithm.')
    parser.add_argument('--eps', type=float, default=.1,
                        help='Value of the entropic regularization parameter.')
    args = parser.parse_args()

    if args.plot:
        plot_benchmark(file_name=args.file)
        raise SystemExit(0)
    else:
        run_benchmark(n_samples=args.n_samples, n_alpha=args.n_alpha,
                      n_epochs=args.n_epochs, step_size=args.step_size,
                      max_layers=args.max_layers, eps=args.eps, gpu=args.gpu,)
