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


BENCH_NAME = "noise_level"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


STYLE = {
    'g1': dict(label=r"$|g_1 - g^*|$", color='C0'),
    'g2': dict(label=r"$|g_2 - g^*|$", color='C1'),
    'g3': dict(label=r"$|g_3 - g^*|$", color='C2'),
}


mem = Memory(location='.', verbose=0)


@mem.cache(ignore=['device'])
def run_one(step, id_rep, n_iters, args, args_star, alpha=None, device=None):
    if step == 'decr':
        assert alpha is not None
        print(f"Starting job for step=decr({alpha:.1e}) [rep={id_rep}]")
        step_ = lambda t: 1 / (1 + t) ** alpha
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


def run_benchmark(n_average=10, random_state=None):
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


def run_benchmark_2(n_average=10, random_state=None):
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
    df.g1 **= 2
    df.g2 **= 2
    df.g3 **= 2
    y = df.groupby('step').median()
    q1, q3 = .25, .75
    y_q1 = df.groupby('step').quantile(q1)
    y_q3 = df.groupby('step').quantile(q3)

    df_decr = pd.read_pickle(file_name_curve)
    df_decr = df_decr[df_decr.step == 'decr']
    n_iters = df_decr.n_iters.mean()

    n_plots = 2
    fig = plt.figure(figsize=(6.4 * n_plots, 7.2))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=n_plots,
                               height_ratios=[.05, .95])
    ax_curve = fig.add_subplot(gs[1, 0])
    ax_noise = fig.add_subplot(gs[1, 1])
    handles = []
    for k, style in STYLE.items():
        handles.append(ax_noise.plot(y.index, y[k], **style)[0])
        ax_noise.fill_between(y.index, y[k] - y_q1[k],
                              y[k] + y_q3[k], alpha=.3,
                              color=style['color'])

        ax_curve.loglog(n_iters, df_decr[k].mean(), **style)
    handles.append(ax_curve.loglog(n_iters, df_decr.z.mean(), 'k--',
                                   linewidth=6)[0])
    # Format ax_curve
    ax_curve.set_title("(a) Decreasing step size")
    ax_curve.set_xscale('log')
    ax_curve.set_xlabel(r'Iteration $t$')
    ax_curve.set_yscale('log')
    ax_curve.set_ylabel(r'')
    ax_curve.set_xlim(min(n_iters), max(n_iters))

    # Format ax_noise
    ax_curve.set_title("(a) Constant step size")
    ax_noise.set_xscale('log')
    ax_noise.set_xlabel(r'Step size $\rho$')
    ax_noise.set_yscale('log')
    ax_noise.set_ylabel(r'Noise level $\sigma^2$')
    ax_noise.set_xlim(y.index.min(), y.index.max())

    # Add legend
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    ax.legend(handles, [r'$|g_1^t - g^*|$', r'$|g_2^t - g^*|$',
                        r'$|g_3^t - g^*|$', r'$|z^t - z^*|$'],
              loc='center', bbox_to_anchor=(0, .95, 1, .05), ncol=4,
              fontsize=18)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}.pdf"),
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
        run_benchmark(n_average=args.n_average, random_state=42)
        run_benchmark_2(n_average=args.n_average, random_state=42)
