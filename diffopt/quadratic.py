import torch
import numpy as np

from diffopt.utils import get_np
from diffopt._base import BaseGradientDescent
from diffopt._base import ALGORITHM


class Quadratic(BaseGradientDescent):

    def _get_default_step(self, x, A, B, C, u, v):
        with torch.no_grad():
            n_dim, _ = B.shape
            L_B = np.linalg.norm(get_np(B), ord=2)
            step = 1 / L_B
        return step

    def _get_z0(self, x, A, B, C, u, v):
        n_samples, _ = x.shape
        n_features, _ = B.shape
        return torch.zeros((n_samples, n_features), device=x.device,
                           dtype=x.dtype)

    def _loss_fn(self, z, x, A, B, C, u, v):
        quad = .5 * (x * torch.matmul(x, A)).sum(axis=-1)
        quad += .5 * (z * torch.matmul(z, B)).sum(axis=-1)
        quad += (z * torch.matmul(x, C)).sum(axis=-1)
        return (quad + torch.matmul(x, u) + torch.matmul(z, v)).sum()

    def _get_grad_z(self, z, x, A, B, C, u, v):
        return torch.matmul(z, B) + torch.matmul(x, C) + v

    def _get_grad_x(self, z, x, A, B, C, u, v, return_loss=False,
                    retain_graph=False, computation=None):

        if computation is None:
            computation = self.gradient_computation

        n_dim, n_features = C.shape
        if computation == 'autodiff':
            x.grad = None
            loss = self._loss_fn(z, x, A, B, C, u, v)
            grad = torch.autograd.grad(loss, x, retain_graph=retain_graph)[0]

        elif computation == 'analytic':
            with torch.no_grad():
                grad = torch.matmul(x, A) + torch.matmul(z, C.t()) + u
                if return_loss:
                    loss = self._loss_fn(z, x, A, B, C, u, v)
        elif computation == 'implicit':
            with torch.no_grad():
                dx = torch.matmul(x, A) + torch.matmul(z, C.t()) + u
                dz = self._get_grad_z(z, x, A, B, C, u, v)
                B_inv = torch.pinverse(B)
                grad = dx - torch.matmul(dz, torch.matmul(C, B_inv).t())

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
        max_layer = 1000
    elif algorithm == 'sgd':
        step = lambda t: 1e-1 / (t+1) ** alpha
        max_layer = 10000
    else:
        raise NotImplementedError(f"algorithm={algorithm} is not implemented")

    n_iters = np.unique(np.logspace(0, np.log10(max_layer), 50, dtype=int))
    log_callbacks = ['z', 'g1', 'g2', 'g3', 'loss']

    # Generate the problem
    n_samples = 30
    n, p = 10, 30
    alpha = .7
    rng = np.random.RandomState(0)
    v = rng.randn(p)
    u = rng.randn(n)
    x = rng.randn(2, n)
    eig_A = np.linspace(.1, 1, n)
    eig_B = np.linspace(.3, 1, p)

    basis_A = np.linalg.svd(rng.randn(n_samples, n), full_matrices=False)
    basis_B = np.linalg.svd(rng.randn(n_samples, p), full_matrices=False)
    A_ = np.dot(basis_A[0], eig_A[:, None] * basis_A[2])
    B_ = np.dot(basis_B[0], eig_B[:, None] * basis_B[2])

    A = np.dot(A_.T, A_)
    B = np.dot(B_.T, B_)
    C = np.dot(A_.T, B_) / 2

    # Compute true minimizer
    quad_star = Quadratic(n_layers=5000, gradient_computation='analytic',
                          algorithm='gd')
    g_star = quad_star.get_grad_x(x, A, B, C, u, v)
    z_star, _ = quad_star.transform(x, A, B, C, u, v)
    l_star = quad_star.score(x, A, B, C, u, v)
    # z_star, J_star, _ = quad_star.transform_with_jacobian(x, A, B, C, u, v)

    quad = Quadratic(n_layers=max(n_iters), algorithm=algorithm, step=step)
    _, log = quad.transform(
        x, A, B, C, u, v, log_iters=n_iters, log_callbacks=log_callbacks,
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
