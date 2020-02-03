import os
import glob
import itertools
import numpy as np
import pandas as pd
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
def run_one(step, id_rep, n_iters, args, args_star, device=None):
    print(f"Starting job for step={step:.1e} [rep={id_rep}]")

    max_layer = max(n_iters)
    logreg = LogReg(n_layers=max_layer, algorithm='sgd', step=step,
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
        'step': step, 'id_rep': id_rep, 'device': device, 'n_iters': n_iters,
        'z': z_diff, 'g1': g1_diff, 'g2': g2_diff, 'g3': g3_diff
        }


def run_benchmark(n_average=10, random_state=None):
    n, p = 50, 30
    reg = 1 / n
    rng = np.random.RandomState(random_state)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    logreg_star = LogReg(n_layers=1000, gradient_computation='analytic',
                         algorithm='gd', device=device,
                         random_state=rng.randint(65000))
    g_star = logreg_star.gradient_x(x, D, reg)
    z_star, J_star = logreg_star.transform_with_jacobian(x, D, reg)

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
    df.to_pickle(os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_{tag}.pkl"))


def plot_benchmark(file_name=None):

    if file_name is None:
        file_pattern = os.path.join(OUTPUT_DIR, f"{BENCH_NAME}_*.pkl")
        file_list = glob.glob(file_pattern)
        file_list.sort()
        file_name = file_list[-1]

    df = pd.read_pickle(file_name)
    df.g1 = [v[0] for v in df.g1]
    df.g2 = [v[0] for v in df.g2]
    df.g3 = [v[0] for v in df.g3]
    df.g1 **= 2
    df.g2 **= 2
    df.g3 **= 2
    y = df.groupby('step').median()
    q1, q3 = .25, .75
    y_err = df.groupby('step').quantile([q1, q3])

    fig, ax = plt.subplots()
    for k, style in STYLE.items():
        ax.plot(y.index, y[k], **style)
        ax.fill_between(y.index, y[k] - y_err[k][:, q1],
                        y[k] + y_err[k][:, q3], alpha=.3,
                        color=style['color'])
    plt.legend(bbox_to_anchor=(-.02, 1.05, 1., .1), ncol=3,
               loc='lower center', fontsize=18, handlelength=1,
               handletextpad=.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'Step size $\rho$')
    ax.set_yscale('log')
    ax.set_ylabel(r'Noise level $\sigma^2$')
    ax.set_xlim(y.index.min(), y.index.max())
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
