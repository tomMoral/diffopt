import torch
import numpy as np

from diffopt.utils import get_np
from diffopt._base import BaseGradientDescent
from diffopt._base import ALGORITHM


class pNorm(BaseGradientDescent):

    def __init__(self, n_layers, p=4, gradient_computation='analytic',
                 step=None, algorithm='gd', random_state=None,
                 name="Logreg", ctx=None, verbose=1, device=None):

        assert p % 2 == 0, f"p needs to be even. Got p={p}."
        self.p = p

        super().__init__(
            n_layers=n_layers, gradient_computation=gradient_computation,
            step=step, algorithm=algorithm, random_state=random_state,
            name=name, ctx=ctx, verbose=verbose, device=device)

    def _get_default_step(self, x, D):
        with torch.no_grad():
            n_dim, _ = D.shape
            D_ = get_np(D)
            L_B = np.linalg.norm(D_.T.dot(D_ ** (self.p - 1)), ord=2)
            step = 1 / L_B
            step = .1
        return step

    def _get_z0(self, x, D):
        n_samples, _ = x.shape
        _, n_features = D.shape
        return torch.zeros((n_samples, n_features), device=x.device,
                           dtype=x.dtype)

    def _loss_fn(self, z, x, D):
        res = x - torch.matmul(z, D.t())
        return (res ** self.p).sum() / self.p

    def _get_grad_z(self, z, x, D):
        res = (torch.matmul(z, D.t()) - x) ** (self.p - 1)
        return torch.matmul(res, D)

    def _get_grad_x(self, z, x, D, return_loss=False,
                    retain_graph=False, computation=None):

        if computation is None:
            computation = self.gradient_computation

        p = self.p
        n_dim, n_features = D.shape
        if computation == 'autodiff':
            x.grad = None
            loss = self._loss_fn(z, x, D)
            grad = torch.autograd.grad(loss, x, retain_graph=retain_graph)[0]

        elif computation == 'analytic':
            with torch.no_grad():
                grad = (x - torch.matmul(z, D.t())) ** (p - 1)
                if return_loss:
                    loss = self._loss_fn(z, x, D)
        elif computation == 'implicit':
            with torch.no_grad():

                res = x - torch.matmul(z, D.t())
                dx = res ** (p - 1)
                dz = self._get_grad_z(z, x, D)

                C = (p-1) * (res ** (p-2))[:, :, None] * D[None]
                H = torch.matmul(C, D.t())

                try:
                    H_inv = torch.pinverse(H)
                except RuntimeError:
                    print("ok")
                    e, v = torch.symeig(H, eigenvectors=True)
                    e_inv = 1 / e
                    e_inv[e < 1e-12] = 0
                    H_inv = torch.matmul(v, e_inv[..., None] *
                                         v.transpose(-1, -2))

                grad = dx - torch.sum(
                    dz[:, None] * torch.matmul(H_inv, -C), axis=-1)

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

    algorithm = args.algorithm
    if algorithm == 'gd':
        step = None
        max_layer = 10000
    elif algorithm == 'sgd':
        step = lambda t: 1e-1 / (t+1) ** alpha  # noqa E731
        max_layer = 100000
    else:
        raise NotImplementedError(f"algorithm={algorithm} is not implemented")

    n_iters = np.unique(np.logspace(0, np.log10(max_layer), 50, dtype=int))
    log_callbacks = ['z', 'g1', 'g2', 'g3', 'loss']

    # Generate the problem
    order = 6
    n, p = 50, 100
    reg = 1 / n
    alpha = .7
    rng = np.random.RandomState(0)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    quad_star = pNorm(n_layers=1000000, p=order, algorithm='gd')
    # g_star = quad_star.get_grad_x(x, D)
    _, log_star = quad_star.transform(x, D, log_iters=[quad_star.n_layers],
                                      log_callbacks=['z', 'loss', 'g1'])
    log_star = log_star[-1]
    z_star, l_star, g_star = log_star['z'], log_star['loss'], log_star['g1']
    # l_star = quad_star.score(x, D)
    # z_star, J_star, _ = quad_star.transform_with_jacobian(x, D)

    quad = pNorm(n_layers=max(n_iters), algorithm=algorithm, step=step)
    _, log = quad.transform(
        x, D, log_iters=n_iters, log_callbacks=log_callbacks,
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

    ratio = g1_list[0] / z_diff[0]
    plt.semilogy(n_iters, g1_list, label=r'$\|g_1^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g2_list, label=r'$\|g_2^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g3_list, label=r'$\|g_3^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, z_diff * ratio, label=r'$\|z^t - z^*\|$', color='k',
                 linewidth=3, linestyle='dashed')
    plt.plot(n_iters, g1_list[0] / n_iters ** ((order - 2) / (order - 1)))
    x_ = plt.xlabel(r'$t$')
    y_ = plt.ylabel('')
    plt.legend()
    plt.show()
