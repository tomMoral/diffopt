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

BENCH_NAME = "wasserstein_barycenter"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


N_INNER_FULL = 500


def wasserstein_barycenter(alphas, C, eps, n_outer, n_inner, gradient,
                           step_size=.1, device=None, meta={}):
    n_samples, n_alpha = alphas.shape
    # Generate the initial barycenter as the uniform distribution
    beta = np.ones(n_alpha) / n_alpha

    alphas, beta, C = check_tensor(alphas, beta, C, device=device)

    sinkhorn = Sinkhorn(n_layers=n_inner, log_domain=False,
                        gradient_computation=gradient, device=device)

    sinkhorn_full = Sinkhorn(
        n_layers=N_INNER_FULL, log_domain=False,
        gradient_computation='analytic', device=device)

    # Warm start the GPU computation
    G_star, loss = sinkhorn_full.gradient_beta(
                    alphas, beta, C, eps, return_loss=True,
                    output_layer=2)

    results = []
    it_loss = np.logspace(0, np.log10(n_outer), n_outer//10, dtype=int)
    t_start = time()
    for it in range(n_outer):
        print(f"{it/n_outer:.1%}".rjust(7, '.') + '\b' * 7,
              end='', flush=True)
        G = sinkhorn._gradient_beta(alphas, beta, C, eps)
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
                g_diff=np.linalg.norm(G_star.ravel()-G.cpu().numpy().ravel()),
                best=n_inner == N_INNER_FULL, **meta
            ))
            t_start = time()

    print("done".rjust(7, '.'))
    return beta, results


def run_benchmark(n_samples=10, n_alpha=100, eps=1, n_outer=300,
                  step_size=.1, max_layers=64, gpu=False):
    """Benchmark for the wasserstein barycenter computation time

    Parameters:
    -----------
    n_samples: int (default: 10)
        Number of distribution to compute the barycenter from.
    n_alpha: int (default: 100)
        Number of point in the support of the distributions.
    eps: float (default: 1)
        Entropy regularization parameter for the Wasserstein distance.
    n_outer: int (default: 300)
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

    meta = dict(n_samples=n_samples, n_alpha=n_alpha, n_beta=n_alpha,
                point_dim=2)
    alphas, _, C, *_ = make_ot(**meta)

    results = []
    max_layers = int(np.log2(max_layers))
    for n_inner in np.logspace(0, max_layers, num=max_layers + 1,
                               base=2, dtype=int):
        for gradient in ['autodiff', 'analytic']:
            print(f"Fitting {gradient}[{n_inner}]:", end='', flush=True)
            beta_star, res = wasserstein_barycenter(
                alphas, C, eps, n_outer=n_outer, n_inner=n_inner,
                step_size=step_size, gradient=gradient, device=device,
                meta=meta)
            results.extend(res)

    print("Fitting optimal barycenter:", end='', flush=True)
    beta_star, res = wasserstein_barycenter(
        alphas, C, eps, n_outer=2 * n_outer, n_inner=N_INNER_FULL,
        step_size=step_size, gradient='analytic', device=device,
        meta=meta)
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
    eps = 1e-14
    min_loss = df_best.loss.min() - eps
    df = df[~df.best]

    list_n_layers = np.unique(df.n_inner)
    n_col = len(list_n_layers)

    # Create a grid of
    fig = plt.figure(figsize=(12.8, 4.8))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=3,
                               height_ratios=[.075, .925],
                               width_ratios=[.49, .49, .02])
    axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

    styles = {'analytic': '-', 'autodiff': ':'}
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
    axes[0].set_xlabel("Outer iteration $q$")
    axes[0].set_ylabel("$F(x_q, z_t(x_q)) - F^*$")
    axes[1].set_xlabel("Time [sec]")
    axes[0].set_ylim(eps, 1e-2)
    axes[1].set_ylim(eps, 1e-2)
    # axes[1].set_ylabel("Outer iteration $q$")

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.legend(ls_legend_handle, ls_legend_label, ncol=2, loc='center')

    # Boundaries are selected between the power of 2 in logscale
    ax_cb = fig.add_subplot(gs[1, 2])
    boundaries = 2 ** (np.arange(-1, 2 * n_col) / 2)[::2]
    mpl.colorbar.ColorbarBase(
            cmap=plt.get_cmap('viridis', n_col),
            norm=mpl.colors.BoundaryNorm(boundaries, n_col),
            ticks=list_n_layers, format='%d', ax=ax_cb,
            label='Inner iterations $t$'
    )
    ax_cb.yaxis.set_label_position("left")

    plt.subplots_adjust(0.08, 0.13, 0.97, 0.97, wspace=.2, hspace=.1)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_loss.pdf"))

    fig = plt.figure()
    min_loss = df_best.loss.min()
    for i, g in enumerate(np.unique(df.gradient)):
        curve = []
        df_g = df[df.gradient == g]
        for t in np.unique(df.n_inner):
            df_gt = df_g[df_g.n_inner == t]
            curve.append((t, df_gt.loss.iloc[-1] - min_loss))
        curve = np.array(curve)
        plt.loglog(curve[:, 0], curve[:, 1], f'C{i}',
                   label=r"$\delta(g_{i+1})$")
        if g == 'analytic':
            plt.loglog(curve[:, 0], curve[:, 1] ** 2 / curve[0, 1], f'C{i}--')
    plt.xlabel(r"Inner iteration $t$")
    plt.ylabel(r"Final optimization error $\delta$")
    plt.legend()
    plt.ylim(1e-12, 1e-7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_delta.pdf"))

    plt.show()


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
    parser.add_argument('--n-outer', type=int, default=2000,
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
                      n_outer=args.n_outer, step_size=args.step_size,
                      max_layers=args.max_layers, eps=args.eps,
                      gpu=args.gpu)
