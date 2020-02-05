import os
import numpy as np
import matplotlib.pyplot as plt


from diffopt.logreg import LogReg


BENCH_NAME = "convergence_jacobian"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--gpu', type=int, default=None,
                        help='If present, use GPU computations')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu is not None else None

    max_layer = 500
    n, p = 10, 30
    reg = 1 / n
    alpha = .5
    rng = np.random.RandomState(42)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    logreg_star = LogReg(n_layers=1000, gradient_computation='analytic',
                         algorithm='gd', device=device)
    logreg_ana = LogReg(n_layers=max_layer, gradient_computation='analytic',
                        algorithm='gd', device=device)
    logreg_auto = LogReg(n_layers=max_layer, gradient_computation='autodiff',
                         algorithm='gd', device=device)

    g_star = logreg_star.get_grad_x(x, D, reg)
    z_star, J_star, _ = logreg_star.transform_with_jacobian(x, D, reg)

    L = (np.linalg.norm(D, ord=2) ** 2 / 4 / n + reg)
    kappa_th = (1 - reg / L)

    n_iters = np.linspace(1, max_layer, 100, dtype=int)

    *_,  log = logreg_ana.transform(
        x, D, reg, log_iters=n_iters)
    z_diff = np.array([np.linalg.norm((rec['z'] - z_star).ravel())
                       for rec in log])
    *_,  log = logreg_ana.transform_with_jacobian(
        x, D, reg, log_iters=n_iters)
    J_diff = np.array([np.linalg.norm((rec['J'] - J_star).ravel())
                       for rec in log])

    print("done".ljust(10))
    z_diff[z_diff == 0] = np.min(z_diff[z_diff > 0])

    from sklearn.linear_model import LinearRegression
    kappa = np.exp(LinearRegression().fit(n_iters[:, None],
                                          np.log(z_diff)).coef_)
    print(kappa, kappa_th)

    plt.figure(f"jacobian - gd")
    z_diff = 1
    plt.semilogy(n_iters, J_diff / z_diff,
                 label=r'$\frac{\|J^t - J^*\|_F}{|z^t - z^*|}$',
                 color='C3', linewidth=3)

    rate_1 = n_iters * kappa ** (n_iters - 1) * abs(z_star).sum() / z_diff
    rate_2 = n_iters * kappa_th ** (n_iters - 1) * abs(z_star).sum() / z_diff
    plt.plot(n_iters, rate_1, 'k', label=r'Estimated $\kappa$')
    plt.plot(n_iters, rate_2, 'g', label=r'Theoretical $\kappa$')
    plt.xlim(1, max_layer)
    x_ = plt.xlabel(r'Iteration $t$')
    y_ = plt.ylabel('')
    plt.legend()
    plt.show()
