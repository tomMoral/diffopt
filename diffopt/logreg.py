import torch
import numpy as np

from diffopt.utils import get_np
from diffopt._compat import AVAILABLE_CONTEXT
from diffopt._base import BaseGradientDescent, ALGORITHM


DEFAULT_CALLBACKS = {'z': lambda z, *_: z}


class LogReg(BaseGradientDescent):
    f"""Sinkhron network for the OT problem

    Parameters
    ----------
    n_layer : int
        Number of layers in the network.
    gradient_computation : str (default: 'autodiff')
        Control how the gradient is computed. The values should be one of
        {{'autodiff', 'analytic'}}.
    step : int or None (default: None)
        Step-size for the algorithm to compute z. If set to None, it will
        be set to:
            - 1 / L for algorithm='gd'
            - 1 / 50L for algorithm='sgd'
    algorithm : str in {{'gd', 'sgd'}}
        Algorithm to compute the optimal point in the logreg.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
    name : str (default: LogReg)
        Name of the model.
    ctx : str or None
        Context to run the network. Can be in {{{AVAILABLE_CONTEXT}}}
    verbose : int (default: 1)
        Verbosity level.
    device : str or None (default: None)
        Device on which the model is implemented. This parameter should be set
        according to the pytorch API (_eg_ 'cpu', 'gpu', 'gpu/1',..).
    """

    def _get_default_step(self, x, D, reg):
        with torch.no_grad():
            n_dim, _ = D.shape
            L_D = np.linalg.norm(get_np(D), ord=2) ** 2 / n_dim
            step = 1 / (L_D / 4 + reg)
        return step

    def _get_z0(self, x, D, reg):
        n_samples, n_dim = x.shape
        n_dim, n_features = D.shape
        return torch.zeros((n_samples, n_features), dtype=D.dtype,
                           device=D.device)

    def _get_args_i(self, sample_slice, D, reg):
        return D[sample_slice], reg

    def _loss_fn(self, z, x, D, reg):
        pen = .5 * torch.sum(z * z)
        res = - x * torch.matmul(z, D.t())
        return torch.log1p(torch.exp(res)).mean(axis=1).sum() + reg * pen

    def _get_grad_z(self, z, x, D, reg):
        n_dim, _ = D.shape
        logit = - x / (1 + torch.exp(x * torch.matmul(z, D.t())))
        return torch.matmul(logit, D) / n_dim + reg * z

    def _get_grad_x(self, z, x, D, reg, return_loss=False, retain_graph=False,
                    computation=None):

        if computation is None:
            computation = self.gradient_computation

        n_dim, n_features = D.shape
        if computation == 'autodiff':
            x.grad = None
            loss = self._loss_fn(z, x, D, reg)
            grad = torch.autograd.grad(loss, x, retain_graph=retain_graph)[0]

        elif computation == 'analytic':
            with torch.no_grad():
                x_hat = torch.matmul(z, D.t())
                grad = - x_hat / (1 + torch.exp(x * x_hat)) / n_dim
                if return_loss:
                    loss = self._loss_fn(z, x, D, reg)
        elif computation == 'implicit':
            with torch.no_grad():
                x_hat = torch.matmul(z, D.t())
                res = x * x_hat
                f_res = torch.exp(res) / (1 + torch.exp(res)) ** 2
                dzz = (
                    torch.matmul(D.t(), (x ** 2 * f_res)[..., None]*D[None]
                                 ) / n_dim
                    + reg * torch.eye(n_features, device=self.device)[None])
                dxz = (
                    (x * x_hat * f_res - 1 / (1 + torch.exp(res)))[..., None]
                    * D[None]) / n_dim
                dx = -x_hat / (1 + torch.exp(x * x_hat)) / n_dim
                dz = (-torch.matmul(x / (1 + torch.exp(x * x_hat)), D) / n_dim
                      + reg * z)
                dzz_inv_dz, _ = torch.solve(dz[..., None], dzz)
                grad = dx - torch.matmul(dxz, dzz_inv_dz)[..., 0]
                if return_loss:
                    loss = self._loss_fn(z, x, D, reg)

        else:
            raise NotImplementedError(
                f"gradient_computation={self.gradient_computation} is not "
                "implemented")
        assert grad.shape == x.shape

        if return_loss:
            return grad, loss
        return grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import argparse
    parser = argparse.ArgumentParser(
        description='Run gradient computation for logistic regression')
    parser.add_argument('--algorithm', '-a', type=str, default='gd',
                        help='Algorithm to compute the gradient from. Should '
                        f'be in {{{ALGORITHM}}}')
    args = parser.parse_args()

    n, p = 50, 100
    reg = 1 / n
    alpha = .7
    rng = np.random.RandomState(0)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    algorithm = args.algorithm
    if algorithm == 'gd':
        step = None
        max_layer = 1000
    elif algorithm == 'sgd':
        step = lambda t: 1e-1 / (t+1) ** alpha
        max_layer = 10000
    else:
        raise NotImplementedError(f"algorithm={algorithm} is not implemented")

    n_iters = np.unique(np.logspace(0, np.log10(max_layer), 50, dtype=int))
    log_callbacks = ['z', 'g1', 'g2', 'g3']

    # Compute true minimizer
    logreg_star = LogReg(n_layers=10000, gradient_computation='analytic',
                         algorithm='gd')
    g_star = logreg_star.get_grad_x(x, D, reg)
    z_star, _ = logreg_star.transform(x, D, reg)
    # z_star, J_star, _ = logreg_star.transform_with_jacobian(x, D, reg)

    logreg = LogReg(n_layers=max(n_iters),
                    gradient_computation='analytic',
                    algorithm=algorithm, step=step)
    _, log = logreg.transform(
        x, D, reg, log_iters=n_iters, log_callbacks=log_callbacks,
        requires_grad=True)

    z_diff = np.array([np.linalg.norm((rec['z'] - z_star).ravel())
                       for rec in log])
    g1_list = np.array([np.linalg.norm((rec['g1'] - g_star).ravel())
                        for rec in log])
    g2_list = np.array([np.linalg.norm((rec['g2'] - g_star).ravel())
                        for rec in log])
    g3_list = np.array([np.linalg.norm((rec['g3'] - g_star).ravel())
                        for rec in log])

    plt.figure()

    plt.semilogy(n_iters, g1_list, label=r'$\|g_1^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g2_list, label=r'$\|g_2^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g3_list, label=r'$\|g_3^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, z_diff, label=r'$\|z^t - z^*\|$', color='k',
                 linewidth=3, linestyle='dashed')
    x_ = plt.xlabel(r'$t$')
    y_ = plt.ylabel('')
    plt.legend()
    plt.show()
