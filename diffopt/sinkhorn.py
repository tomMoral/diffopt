import torch
import numpy as np

from adopty.utils import check_tensor
from adopty._compat import AVAILABLE_CONTEXT


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
                 ctx=None, verbose=1, device=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        self.name = name
        self._ctx = ctx
        self.device = device
        self.verbose = verbose

        self.tol = tol
        self.n_layers = n_layers
        self.log_domain = log_domain
        self.gradient_computation = gradient_computation

        super().__init__()

    def forward(self, alpha, beta, C, eps, output_layer=None):

        n_alpha, n_beta = C.shape

        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError("Requested output from out-of-bound layer "
                             "output_layer={} (n_layers={})"
                             .format(output_layer, self.n_layers))
        if self.log_domain:
            g = torch.zeros_like(beta)
        else:
            v = torch.ones_like(beta)
            K = torch.exp(- C / eps)

        # Compute the following layers
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
        if not self.log_domain:
            f = eps * torch.log(u)
            g = eps * torch.log(v)

        return f, g

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
                reg = torch.exp(((-C + f.unsqueeze(-1) + g.unsqueeze(-2)) / eps
                                 ).logsumexp((-1, -2))).sum()
            else:
                K = torch.exp(- C / eps)
                reg = torch.sum(torch.exp(f / eps) *
                                torch.matmul(torch.exp(g / eps), K.t()))

            output = torch.sum(f * alpha)
            output += torch.sum(g * beta)
            output -= eps * reg
        return output

    def score(self, alpha, beta, C, eps, output_layer=None, primal=False):
        """Compute the loss for the network's output

        Parameters
        ----------
        alpha : ndarray, shape (n_samples_1)
            First input distribution.
        beta: ndarray, shape (n_samples_2)
            Second input distribution.
        C : ndarray, shape (n_samples_1, n_samples_2)
            Cost matrix between the samples of each distribution.
        eps : float
            Entropic regularization parameter
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. Ifs set to None, output the network's last
            layer.
        primal : boolean (default: False)
            If set to True, output the primal loss function. Else, output the
            dual loss.

        Return
        ------
        loss : float
            Optimal transport loss between alpha, beta, for the given C and eps
        """
        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)
        with torch.no_grad():
            f, g = self(alpha, beta, C, eps, output_layer=output_layer)
            return self._loss_fn(f, g, alpha, beta, C, eps, primal=primal
                                 ).cpu().numpy()

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
                f, g = self(alpha, beta, C, eps, output_layer=output_layer + 1)
                loss.append(self._loss_fn(f, g, alpha, beta, C, eps,
                                          primal=primal).cpu().numpy())
        return np.array(loss)

    def _gradient_beta(self, alpha, beta, C, eps, output_layer=None,
                       return_loss=False):

        if self.gradient_computation == 'autodiff':
            beta = check_tensor(beta, device=self.device, requires_grad=True)
            beta.grad = None

            f, g = self(alpha, beta, C, eps, output_layer=output_layer)
            loss = self._loss_fn(f, g, alpha, beta, C, eps, primal=False)
            loss.backward()
            grad = beta.grad

        elif self.gradient_computation == 'analytic':
            with torch.no_grad():
                f, grad = self(alpha, beta, C, eps, output_layer=output_layer)
                if return_loss:
                    loss = self._loss_fn(f, grad, alpha, beta, C, eps,
                                         primal=False)
        if grad.shape != beta.shape:
            assert beta.dim() == 1
            grad = grad.sum(axis=0)

        if return_loss:
            return grad, loss
        return grad

    def gradient_beta(self, alpha, beta, C, eps, output_layer=None,
                      return_loss=False):
        """Compute the gradient of Sinkhorn relative to beta with autodiff."""
        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)
        res = self._gradient_beta(alpha, beta, C, eps,
                                  output_layer=output_layer,
                                  return_loss=return_loss)
        if return_loss:
            return res[0].cpu().numpy(), res[1].cpu().numpy()
        return res.cpu().numpy()

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
        f, g = self(alpha, beta, C, eps, output_layer=output_layer)
        return torch.autograd.grad(
            g, beta, grad_outputs=torch.eye(n_features))[0].cpu().numpy()

    def transform(self, alpha, beta, C, eps, output_layer=None):
        """Compute the dual variables associate to the transport plan.

        The transport plan can be recovered using the formula:
            P = exp(f / eps)[:, None] * exp(-C / eps) * exp (g / eps)[None]
        """
        # Compat numpy
        alpha, beta, C = check_tensor(alpha, beta, C, device=self.device)

        with torch.no_grad():
            f, g = self(alpha, beta, C, eps, output_layer=output_layer)

        return f.cpu().numpy(), g.cpu().numpy()


if __name__ == "__main__":

    from adopty.datasets.optimal_transport import make_ot

    eps = .1
    a, b, C, *_ = make_ot(n_alpha=10, n_beta=30, n_samples=500)

    n_iters = np.arange(1, 100)
    # Compute true minimizer
    sinkhorn_ana = Sinkhorn(n_layers=1000, log_domain=False,
                            gradient_computation='analytic')
    sinkhorn_auto = Sinkhorn(n_layers=1000, log_domain=False,
                             gradient_computation='autodiff')
    g_star = sinkhorn_ana.gradient_beta(a, b, C, eps)
    f_, g_ = sinkhorn_ana.transform(a, b, C, eps)
    g1_list = []
    g2_list = []
    z_diff = []
    for n_iter in n_iters:
        print(f"{n_iter/ n_iters.max():7.1%}\r", end='', flush=True)
        f, g = sinkhorn_ana.transform(a, b, C, eps)
        diff = np.sqrt(np.linalg.norm(f - f_) ** 2
                       + np.linalg.norm(g - g_) ** 2)
        z_diff.append(diff)
        g1 = sinkhorn_ana.gradient_beta(a, b, C, eps, n_iter)
        g2 = sinkhorn_auto.gradient_beta(a, b, C, eps, n_iter)
        g1_list.append(np.linalg.norm((g1 - g_star).ravel()))
        g2_list.append(np.linalg.norm((g2 - g_star).ravel()))
    print("done".ljust(10))

    g1_list, g2_list = np.array(g1_list), np.array(g2_list)
    import matplotlib.pyplot as plt
    plt.semilogy(n_iters, g1_list, label=r'$\|g_1^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g2_list, label=r'$\|g_2^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, z_diff, label=r'$\|z^t - z^*\|$', color='k',
                 linewidth=3, linestyle='dashed')
    x_ = plt.xlabel(r'$t$')
    y_ = plt.ylabel('')
    plt.legend()
    plt.show()
