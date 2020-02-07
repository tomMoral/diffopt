import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
from datetime import datetime
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, Memory


from diffopt.logreg import LogReg
from diffopt.utils.viz import make_legend, STYLES


BENCH_NAME = "noise_level"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


mem = Memory(location='.', verbose=0)


@mem.cache(ignore=['device'])
def run_one(step, id_rep, n_iters, args, args_star, alpha=None, device=None):
    if step == 'decr':
        assert alpha is not None
        print(f"Starting job for step=decr({alpha:.1e}) [rep={id_rep}]")
        step_ = lambda t: 1 / (1 + t) ** alpha  # noqa E731
    else:
        step_ = step
        print(f"Starting job for step={step:.1e} [rep={id_rep}]")

    max_layer = max(n_iters)
    logreg = LogReg(n_layers=max_layer, algorithm='sgd', step=step_,
                    device=device)

    z, log = logreg.transform(
        *args, log_iters=n_iters, log_callbacks={
            'z': lambda z, *_: z,
            'g1': lambda *args: logreg._gradient_x(
                *args, computation='analytic'),
            'g2': lambda *args: logreg._gradient_x(
                *args, computation='autodiff', retain_graph=True),
            'g3': lambda *args: logreg._gradient_x(
                *args, computation='implicit')
        }, requires_grad=True)

    # log contrain
    z_star, g_star = args_star
    z_diff = np.array([np.linalg.norm((rec['z'] - z_star).ravel())
                       for rec in log])
    g1_diff = np.array([np.linalg.norm((rec['g1'] - g_star).ravel())
                        for rec in log])
    g2_diff = np.array([np.linalg.norm((rec['g2'] - g_star).ravel())
                        for rec in log])
    g3_diff = np.array([np.linalg.norm((rec['g3'] - g_star).ravel())
                        for rec in log])

    return {
        'step': step, 'alpha': alpha, 'id_rep': id_rep, 'device': device,
        'n_iters': n_iters, 'z': z_diff, 'g1': g1_diff, 'g2': g2_diff,
        'g3': g3_diff
        }


def run_benchmark_noise(n_average=10, random_state=None):
    n, p = 50, 30
    reg = 1 / n
    rng = np.random.RandomState(random_state)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    logreg_star = LogReg(n_layers=1000, gradient_computation='analytic',
                         algorithm='gd', device=device)
    star = logreg_star.transform(
        x, D, reg, log_iters=[logreg_star.n_layers],
        log_callbacks=['z', 'g1'])[1][-1]
    z_star, g_star = star['z'], star['g1']

    # Check optimality
    logit = - x / (1 + np.exp(x * np.dot(z_star, D.T)))
    grad_z = np.dot(logit, D) / n + reg * z_star
    print(np.linalg.norm(grad_z, ord=2, axis=-1))

    max_layer = 500000

    # n_iters = np.unique(np.logspace(0, np.log10(max_layer), 25, dtype=int))
    n_iters = [max_layer]
    kwargs = dict(args=(x, D, reg), args_star=(z_star, g_star),
                  n_iters=n_iters, device=device)

    steps = np.logspace(-3, -1, 20)
    it_parallel = itertools.product(steps, range(n_average))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one)(step=step, id_rep=id_rep, **kwargs)
        for step, id_rep in it_parallel
    )
    print("\rBench: done".ljust(40))
    df = pd.DataFrame(results)
    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_noise_{tag}.pkl"))


def run_benchmark_curve(n_average=10, random_state=None):
    n, p = 50, 30
    reg = 1 / n
    alpha = .8
    rng = np.random.RandomState(random_state)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    logreg_star = LogReg(n_layers=1000, gradient_computation='analytic',
                         algorithm='gd', device=device)
    star = logreg_star.transform(
        x, D, reg, log_iters=[logreg_star.n_layers],
        log_callbacks=['z', 'g1'])[1][-1]
    z_star, g_star = star['z'], star['g1']

    # Check optimality
    logit = - x / (1 + np.exp(x * np.dot(z_star, D.T)))
    grad_z = np.dot(logit, D) / n + reg * z_star
    print(np.linalg.norm(grad_z, ord=2, axis=-1))

    max_layer = 500000

    n_iters = np.unique(np.logspace(0, np.log10(max_layer), 50, dtype=int))
    kwargs = dict(args=(x, D, reg), args_star=(z_star, g_star),
                  n_iters=n_iters, device=device, alpha=alpha)

    steps = [.02, 'decr']
    it_parallel = itertools.product(steps, range(n_average))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one)(step=step, id_rep=id_rep, **kwargs)
        for step, id_rep in it_parallel
    )
    print("\rBench: done".ljust(40))
    df = pd.DataFrame(results)
    tag = f"{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_curve_{tag}.pkl"))


def plot_gradient_estimation(df, ax, title=None):
    n_iters = df.n_iters.mean()
    for k, style in STYLES.items():
        k_diff = np.array([v for v in df[k]])
        quantiles = np.quantile(k_diff, [.1, .9], axis=0)
        ax.loglog(n_iters, k_diff.mean(axis=0), **style)
        ax.fill_between(n_iters, quantiles[0], quantiles[1],
                        color=style['color'], alpha=.3)

    # Format ax_curve
    ax.set_title(title)
    ax.set_xscale('log')
    ax.set_xlabel(r'Iteration $t$')
    ax.set_yscale('log')
    ax.set_ylabel(r'')
    ax.set_xlim(1e1, n_iters.max())


def plot_benchmark(file_name=None):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_noise_*.pkl")
        file_list = glob.glob(file_pattern)
        file_list.sort()
        file_name_noise = file_list[-1]
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_curve_*.pkl")
        file_list = glob.glob(file_pattern)
        file_list.sort()
        file_name_curve = file_list[-1]

    df = pd.read_pickle(file_name_noise)
    df = df.drop(['alpha', 'device', 'n_iters'], axis=1)
    df.g1 = [v[0] for v in df.g1]
    df.g2 = [v[0] for v in df.g2]
    df.g3 = [v[0] for v in df.g3]
    df.z = [v[0] for v in df.z]
    y = df.groupby('step').mean()
    q1, q3 = .1, .9
    y_q1 = df.groupby('step').quantile(q1)
    y_q3 = df.groupby('step').quantile(q3)

    df_curve = pd.read_pickle(file_name_curve)
    steps, alpha = set(df_curve.step), df_curve.alpha.iloc[0]
    # results = []
    # for id_rep, rec in df_curve.iterrows():
    #     for it, g1, g2, g3, z in zip(rec['n_iters'], rec['g1'], rec['g2'],
    #                                  rec['g3'], rec['z']):
    #         results.append(
    #             dict(step=rec['step'], alpha=rec['alpha'],
    #                  device=rec['device'], id_rep=id_rep,
    #                  g1=g1, g2=g2, g3=g3, z=z, it=it)
    #         )

    # df = pd.DataFrame(results)

    n_plots = 2
    fig = plt.figure(figsize=(6.4 * n_plots, 7.2))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=n_plots,
                               height_ratios=[q1, q3])
    ax_curve = fig.add_subplot(gs[1, 1])
    ax_noise = fig.add_subplot(gs[1, 0])
    for k, style in STYLES.items():
        ax_noise.plot(y.index, y[k], **style)[0]
        ax_noise.fill_between(y.index, y_q1[k],
                              y_q3[k], alpha=.3,
                              color=style['color'])

    # Format ax_noise
    ax_noise.set_title("(a) Noise Level for\nConstant step size")
    ax_noise.set_xscale('log')
    ax_noise.set_xlabel(r'Step size $\rho$')
    ax_noise.set_yscale('log')
    ax_noise.set_ylabel(r'Noise level')
    ax_noise.set_xlim(y.index.min(), y.index.max())

    plot_gradient_estimation(df_curve[df_curve.step == 'decr'], ax_curve,
                             title="(b) Decreasing step size\n$\\rho_t = "
                             f"t^{{-\\alpha}}$; $\\alpha = {alpha}$")
    ax_curve.set_ylabel(r"$\mathbb E\left[|g^i_t - g^*|\right]$")

    # Add legend
    ax_legend = fig.add_subplot(gs[0, :])
    make_legend(ax_legend)
    plt.subplots_adjust(left=.08, bottom=.13, right=.99, top=.96,
                        wspace=.22, hspace=.24)

    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}.pdf"),
                bbox_inches='tight', pad_inches=0)

    n_plots = len(steps)
    fig = plt.figure(figsize=(6.4 * n_plots, 7.2))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=n_plots,
                               height_ratios=[.05, .95])
    for i, step in enumerate(steps):
        ax = fig.add_subplot(gs[1, i])
        title = (
            'Decreasing step size\n$\\rho_t = t^{-\\alpha}$; '
            f'$\\alpha = {alpha}$' if step == 'decr'
            else f'Constant step size\n$\\rho = {step}$')
        plot_gradient_estimation(
            df_curve[df_curve.step == step], ax,
            title=f"({chr(97 + i)}) {title}")

        ax.set_ylabel(r"$\mathbb E\left[|g^i_t - g^*|\right]$")
    ax_legend = fig.add_subplot(gs[0, :])

    make_legend(ax_legend)
    plt.subplots_adjust(left=.08, bottom=.1, right=.99, top=.96,
                        wspace=.22, hspace=.3)

    plt.savefig(os.path.join(OUTPUT_DIR, f"sgd_gradient_estimation.pdf"),
                bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', type=int, default=None,
                        help='If present, use GPU computations')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='# of CPU to use for computation')
    parser.add_argument('--n-average', type=int, default=10,
                        help='# repetition to compute the average')
    parser.add_argument('--plot', action='store_true',
                        help='Show the results from the benchmark')
    parser.add_argument('--file', type=str, default=None,
                        help='File to plot')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu is not None else None

    if args.plot:
        plot_benchmark(args.file)
    else:
        run_benchmark_noise(n_average=args.n_average, random_state=42)
        run_benchmark_curve(n_average=args.n_average, random_state=42)
