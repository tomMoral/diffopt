import torch
import numpy as np
from contextlib import nullcontext

from diffopt._base import GRADIENTS
from diffopt._compat import AVAILABLE_CONTEXT
from diffopt.utils import check_tensor, get_np


CALLBACKS = {
    'z': lambda model, f, g, *_: g,
    'g1': lambda model, *args: model._get_grad_beta(
        *args, computation='analytic'),
    'g2': lambda model, *args: model._get_grad_beta(
        *args, computation='autodiff', retain_graph=True),
    'g3': lambda model, *args: model._get_grad_beta(
        *args, computation='implicit'),
    'loss': lambda model, *args: model._loss_fn(*args),
    'J': lambda model, z, x, *_: model._get_jabobian_zx(
        z, x, retain_graph=True),
    }
DEFAULT_CALLBACKS = ['z']


def log_dot_exp(A, b, eps):
    """Compute the dot product between exp(A) and exp(b) in log domain.

    Parameters
    ----------
    A : torch.Tensor, shape (n_alpha, n_beta)
        Matrix used for the matrix vector product in log domain
    b : torch.Tensor, shape (n_beta,) or (n_samples, n_beta)
    """
    if b.dim() == 1:
        return ((-A + b.unsqueeze(-2)) / eps).logsumexp(axis=-1)
    assert b.dim() == 2
    return ((-A.unsqueeze(0) + b.unsqueeze(-2)) / eps).logsumexp(axis=-1)


class Sinkhorn(torch.nn.Module):
    f"""Sinkhron network for the OT problem

    Parameters
    ----------
    n_layer : int
        Number of layers in the network.
    tol : float
        Stopping criterion. When the dual variable move less than the specified
        tolerance, return from the sinkhorn algorithm. If set to None, run for
        as many iteration as requested.
    log_domain: bool (default: True)
        If set to True, run the computation in the log-domain. This is useful
        for small values of eps but might slow down the computations.
    gradient_computation : str (default: 'autodiff')
        Control how the gradient is computed. The values should be one of
        {{'autodiff', 'analytic'}}.
    name : str (default: Sinkhorn)
        Name of the model.
    ctx : str or None
        Context to run the network. Can be in {{{AVAILABLE_CONTEXT}}}
    verbose : int (default: 1)
        Verbosity level.
    device : str or None (default: None)
        Device on which the model is implemented. This parameter should be set
        according to the pytorch API (_eg_ 'cpu', 'gpu', 'gpu/1',..).
    """

    def __init__(self, n_layers, tol=None, log_domain=True,
                 gradient_computation='autodiff', name="Sinkhorn",
                 ctx=None, verbose=1, device=None, random_state=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        assert gradient_computation in GRADIENTS

        self.tol = tol
        self.n_layers = n_layers
        self.log_domain = log_domain
        self.random_state = random_state
        self.gradient_computation = gradient_computation

        self.name = name
        self._ctx = ctx
        self.device = device
        self.verbose = verbose

        super().__init__()

    def forward(self, alpha, beta, C, eps, output_layer=None, log_iters=None,
                log_callbacks=DEFAULT_CALLBACKS):

        n_alpha, n_beta = C.shape

        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError("Requested output from out-of-bound layer "
                             "output_layer={} (n_layers={})"
                             .format(output_layer, self.n_layers))

        if log_iters is None:
            log_iters = [output_layer]

        if self.log_domain:
            g = torch.zeros_like(beta)
        else:
            v = torch.ones_like(beta)
            K = torch.exp(- C / eps)

        # Compute the following layers
        log = []
        for id_layer in range(output_layer):
            if self.log_domain:
                g_hat = g
                f = eps * (torch.log(alpha) - log_dot_exp(C, g, eps))
                g = eps * (torch.log(beta) - log_dot_exp(C.t(), f, eps))
            else:
                v_hat = v
                u = alpha / torch.matmul(v, K.t())
                v = beta / torch.matmul(u, K)

            # Check if the variables are not moving anymore.
            if self.tol is not None and id_layer % 10 == 0:
                if self.log_domain:
                    err = torch.norm(g - g_hat)
                else:
                    err = torch.norm(v - v_hat)
                if err < 1e-10:
                    break

            if (id_layer + 1) % 100 == 0:
                print(f"{(id_layer + 1) / output_layer:6.1%}" + '\b'*6,
                      end='', flush=True)
            if id_layer + 1 in log_iters:
                if not self.log_domain:
                    f, g = eps * torch.log(u), eps * torch.log(v)
                rec = {k: get_np(CALLBACKS[k](self, f, g, alpha,
                                              beta, C, eps))
                       for k in log_callbacks}
                rec['iter'] = id_layer
                log.append(rec)

        if not self.log_domain:
            f, g = eps * torch.log(u), eps * torch.log(v)

        if log_iters is not None:
            return f, g, log
        return f, g, None

    def _loss_fn(self, f, g, alpha, beta, C, eps, primal=False):
        if primal:
            K = torch.exp(- C / eps)
            if self.log_domain:
                P = torch.exp((-C + f.unsqueeze(-1) + g.unsqueeze(-2)) / eps)
                output = torch.dot(P.view(-1), C.view(-1))
            else:
                u = torch.exp(f / eps)
                v = torch.exp(g / eps)
                K = torch.exp(- C / eps)
                P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
                output = torch.dot(P.view(-1), C.view(-1))
        else:
            if self.log_domain:
                cross = torch.exp(((-C + f.unsqueeze(-1) + g.unsqueeze(-2))
                                  / eps).logsumexp((-1, -2))).sum()
            else:
                K = torch.exp(- C / eps)
                cross = torch.sum(torch.exp(f / eps)
                                  * torch.matmul(torch.exp(g / eps), K.t()))

            output = torch.sum(f * alpha)
            output += torch.sum(g * beta)
            output -= eps * cross
        return output

    def score(self, alpha, beta, C, eps, primal=False, output_layer=None):
        """Compute the loss for the network's output

        Parameters
        ----------
        alpha : ndarray, shape (n_samples, n_alpha)
            First input distribution.
        beta: ndarray, shape (n_beta,)
            Second input distribution.
        C : ndarray, shape (n_alpha, n_beta)
            Cost matrix between the samples of each distribution.
        eps : float
            Entropic regularization parameter
        primal : boolean (default: False)
            If set to True, output the primal loss function. Else, output the
            dual loss.
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. Ifs set to None, output the network's last
            layer.

        Return
        ------
        loss : float
            Regularized logreg loss between x and Dz, with regularization reg
        """
        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)
        with torch.no_grad():
            f, g, _ = self(alpha, beta, C, eps, output_layer=output_layer)
            return get_np(self._loss_fn(f, g, alpha, beta, C, eps,
                                        primal=primal))

    def compute_loss(self, alpha, beta, C, eps, primal=False):
        """Compute the loss  along the network's layers

        Parameters
        ----------
        alpha : ndarray, shape (n_alpha,)
            First input distribution.
        beta: ndarray, shape (n_beta,)
            Second input distribution.
        C : ndarray, shape (n_alpha, n_beta)
            Cost matrix between the samples of each distribution.
        eps : float
            Entropic regularization parameter
        primal : boolean (default: False)
            If set to True, output the primal loss function. Else, output the
            dual loss.
        """
        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)
        loss = []
        with torch.no_grad():
            for output_layer in range(self.n_layers):
                f, g, _ = self(alpha, beta, C, eps,
                               output_layer=output_layer + 1)
                loss.append(get_np(self._loss_fn(f, g, alpha, beta, C, eps,
                                                 primal=primal)))
        return np.array(loss)

    def _get_grad_beta(self, f, g, alpha, beta, C, eps, return_loss=False,
                       retain_graph=False, computation=None):

        if computation is None:
            computation = self.gradient_computation

        if computation == 'autodiff':
            beta = check_tensor(beta, device=self.device, requires_grad=True)
            beta.grad = None

            loss = self._loss_fn(f, g, alpha, beta, C, eps, primal=False)
            grad = torch.autograd.grad(
                loss, beta, retain_graph=retain_graph)[0]

        elif computation == 'analytic':
            with torch.no_grad():
                grad = g
                if return_loss:
                    loss = self._loss_fn(f, grad, alpha, beta, C, eps,
                                         primal=False)
        elif computation == 'implicit':
            with torch.no_grad():
                n_samples, _ = alpha.shape
                n, m = C.shape
                z = torch.zeros((n_samples, n + m), device=alpha.device)
                H = torch.zeros((n_samples, n+m, n+m), device=alpha.device)

                dx = g
                u, v = torch.exp(f / eps), torch.exp(g / eps)
                K = torch.exp(-C / eps)
                P = u[:, :, None] * K[None] * v[:, None]
                z[:, :n] = alpha - u * torch.matmul(v, K.t())
                z[:, n:] = beta - v * torch.matmul(u, K)
                bias = z.sum(axis=-1, keepdims=True)
                z -= bias / (n + m)

                # Compute the Hessian zz and solve h_inv . z
                H[:, :n, :n] = torch.diag_embed(-P.sum(axis=-1)/eps)
                H[:, n:, n:] = torch.diag_embed(-P.sum(axis=-2)/eps)
                H[:, :n, n:] = -P / eps
                H[:, n:, :n] = -P.transpose(-2, -1) / eps
                e, v = torch.symeig(H, eigenvectors=True)
                e_inv = 1 / e
                e_inv[e > -1e-12] = 0
                H_inv = torch.matmul(v, e_inv[..., None] * v.transpose(-1, -2))
                dz = (H_inv * z[:, None, :]).sum(axis=-1)[:, n:]
                grad = dx - dz
                if return_loss:
                    loss = self._loss_fn(f, g, alpha, beta, C, eps,
                                         primal=False)
        if grad.shape != beta.shape:
            assert beta.dim() == 1
            grad = grad.sum(axis=0)

        if return_loss:
            return grad, loss
        return grad

    def gradient_beta(self, alpha, beta, C, eps, output_layer=None,
                      return_loss=False, computation=None):
        """Compute the gradient of Sinkhorn relative to beta with autodiff."""
        if computation is None:
            computation = self.gradient_computation

        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)
        if computation == 'autodiff':
            beta = check_tensor(beta, device=self.device, requires_grad=True)
        f, g, _ = self(alpha, beta, C, eps, output_layer=output_layer)
        res = self._get_grad_beta(f, g, alpha, beta, C, eps,
                                  return_loss=return_loss)
        if return_loss:
            return get_np(res[0]), get_np(res[1])
        return get_np(res)

    def get_grad_x(self, *args, **kwargs):
        return self.gradient_beta(*args, **kwargs)

    def get_jacobian_beta(self, alpha, beta, C, eps, output_layer=None):
        """Compute the Jacobian of the scale dual variable g relative to beta.
        """
        n_features = beta.shape

        alpha = check_tensor(alpha, device=self.device)
        beta = check_tensor(beta, device=self.device, require_grad=True)
        C = check_tensor(C, device=self.device)

        # Contruct the matrix to probe the jacobian
        beta = beta.squeeze()
        beta = beta.repeat(n_features, 1)
        f, g, _ = self(alpha, beta, C, eps, output_layer=output_layer)
        return get_np(torch.autograd.grad(
            g, beta, grad_outputs=torch.eye(n_features))[0])

    def transform(self, alpha, beta, C, eps, output_layer=None, log_iters=None,
                  log_callbacks=DEFAULT_CALLBACKS, requires_grad=False):
        """Compute the dual variables associate to the transport plan.

        The transport plan can be recovered using the formula:
            P = exp(f / eps)[:, None] * exp(-C / eps) * exp (g / eps)[None]
        """
        # Compat numpy
        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)
        beta = check_tensor(beta, requires_grad=True)

        with nullcontext() if requires_grad else torch.no_grad():
            f, g, log = self(alpha, beta, C, eps, output_layer=output_layer,
                             log_iters=log_iters, log_callbacks=log_callbacks)

        return (get_np(f), get_np(g)), log


if __name__ == "__main__":

    from adopty.datasets.optimal_transport import make_ot

    eps = .1
    a, b, C, *_ = make_ot(n_alpha=10, n_beta=30, n_samples=500)
    a = a[:]

    n_iters = np.arange(1, 100, 3)
    # Compute true minimizer
    sinkhorn_ana = Sinkhorn(n_layers=1000, log_domain=False,
                            gradient_computation='analytic')
    sinkhorn_auto = Sinkhorn(n_layers=1000, log_domain=False,
                             gradient_computation='autodiff')
    sinkhorn_impl = Sinkhorn(n_layers=1000, log_domain=False,
                             gradient_computation='implicit')
    g_star = sinkhorn_ana.gradient_beta(a, b, C, eps)
    (f_, g_), _ = sinkhorn_ana.transform(a, b, C, eps)
    z_diff = []
    g1_list, g2_list, g3_list = [], [], []
    for n_iter in n_iters:
        print(f"{n_iter/ n_iters.max():7.1%}\r", end='', flush=True)
        (f, g), _ = sinkhorn_ana.transform(a, b, C, eps)
        diff = np.sqrt(np.linalg.norm(f - f_) ** 2
                       + np.linalg.norm(g - g_) ** 2)
        z_diff.append(diff)
        g1 = sinkhorn_ana.gradient_beta(a, b, C, eps, n_iter)
        g2 = sinkhorn_auto.gradient_beta(a, b, C, eps, n_iter)
        g3 = sinkhorn_impl.gradient_beta(a, b, C, eps, n_iter)
        g1_list.append(np.linalg.norm((g1 - g_star).ravel()))
        g2_list.append(np.linalg.norm((g2 - g_star).ravel()))
        g3_list.append(np.linalg.norm((g3 - g_star).ravel()))
    print("done".ljust(10))

    g1_list, g2_list = np.array(g1_list), np.array(g2_list)
    import matplotlib.pyplot as plt
    plt.semilogy(n_iters, g1_list, label=r'$\|g_1^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g2_list, label=r'$\|g_2^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g3_list, label=r'$\|g_3^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, z_diff, label=r'$\|z^t - z^*\|$', color='k',
                 linewidth=3, linestyle='dashed')
    x_ = plt.xlabel(r'$t$')
    y_ = plt.ylabel('')
    plt.legend()
    plt.show()
