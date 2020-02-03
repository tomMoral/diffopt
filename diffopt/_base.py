import torch
import numpy as np
from abc import abstractmethod
from contextlib import nullcontext

from diffopt._compat import AVAILABLE_CONTEXT
from diffopt.utils import check_random_state
from diffopt.utils import check_tensor, get_np


CALLBACKS = {
    'z': lambda model, z, *_: z,
    'g1': lambda model, *args: model._get_grad_x(
        *args, computation='analytic'),
    'g2': lambda model, *args: model._get_grad_x(
        *args, computation='autodiff', retain_graph=True),
    'g3': lambda model, *args: model._get_grad_x(
        *args, computation='implicit'),
    'loss': lambda model, *args: model._loss_fn(*args),
    'J': lambda model, z, x, *_: model._get_jabobian_zx(
        z, x, retain_graph=True),
    }
DEFAULT_CALLBACKS = ['z']
ALGORITHM = ['gd', 'sgd']
GRADIENTS = ['analytic', 'autodiff', 'implicit']


class BaseGradientDescent(torch.nn.Module):
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

    def __init__(self, n_layers, gradient_computation='autodiff', step=None,
                 algorithm='gd', random_state=None, name="Logreg", ctx=None,
                 verbose=1, device=None):
        if ctx:
            msg = "Context {} is not available on this computer."
            assert ctx in AVAILABLE_CONTEXT, msg.format(ctx)
        else:
            ctx = AVAILABLE_CONTEXT[0]

        assert algorithm in ALGORITHM
        assert gradient_computation in GRADIENTS

        self.step = step
        self.n_layers = n_layers
        self.algorithm = algorithm
        self.random_state = random_state
        self.gradient_computation = gradient_computation

        self.name = name
        self._ctx = ctx
        self.device = device
        self.verbose = verbose

        super().__init__()

    def forward(self, x, *loss_args, output_layer=None, log_iters=None,
                log_callbacks=DEFAULT_CALLBACKS):

        rng = check_random_state(self.random_state)

        if output_layer is None:
            output_layer = self.n_layers
        elif output_layer > self.n_layers:
            raise ValueError("Requested output from out-of-bound layer "
                             "output_layer={} (n_layers={})"
                             .format(output_layer, self.n_layers))
        if log_iters is None:
            log_iters = [output_layer]

        # Init the first layer ot zero
        z = self._get_z0(x, *loss_args)

        # compute a safe step size
        if self.step is None:
            step = self._get_default_step(x, *loss_args)
            get_step = lambda i: step  # noqa E371
        elif not callable(self.step):
            # to use a constant step size
            get_step = lambda i: self.step  # noqa E371
        else:
            get_step = self.step

        # Compute a gradient descent
        log = []
        n_dim = x.shape[1]
        id_samples = rng.randint(n_dim, size=output_layer)
        for id_layer in range(output_layer):
            if self.algorithm == 'gd':
                grad_z = self._get_grad_z(z, x, *loss_args)
            elif self.algorithm == 'sgd':
                id_sample = id_samples[id_layer]
                slice_sample = slice(id_sample, id_sample+1)
                xi = x[:, slice_sample]
                loss_args_i = self._get_args_i(slice_sample, *loss_args)
                grad_z = self._get_grad_z(z, xi, *loss_args_i)
            else:
                raise NotImplementedError(
                    f"algorithm={self.algorithm} is not implemented")
            step_i = get_step(id_layer)
            z = z - step_i * grad_z

            if (id_layer + 1) % 100 == 0:
                print(f"{(id_layer + 1) / output_layer:6.1%}" + '\b'*6,
                      end='', flush=True)
            if id_layer + 1 in log_iters:
                log.append({k: get_np(CALLBACKS[k](self, z, x, *loss_args))
                            for k in log_callbacks})

        if log_iters is not None:
            return z, log
        return z, None

    @abstractmethod
    def _get_default_step(self, x, *loss_args):
        ...

    @abstractmethod
    def _get_grad_z(self, z, x, *loss_args):
        ...

    @abstractmethod
    def _loss_fn(self, z, x, *loss_args):
        ...

    @abstractmethod
    def _get_grad_x(self, z, x, *loss_args, return_loss=False,
                    retain_graph=False, computation=None):
        ...

    def score(self, x, *loss_args, output_layer=None):
        """Compute the loss for the network's output

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            Data to approximate.
        D: ndarray, shape (n_dim, n_features)
            Design matrix for the problem.
        reg : float
            L2 regularization parameter
        output_layer : int (default: None)
            Layer to output from. It should be smaller than the number of
            layers of the network. Ifs set to None, output the network's last
            layer.

        Return
        ------
        loss : float
            Regularized logreg loss between x and Dz, with regularization reg
        """
        x, *loss_args = check_tensor(x, *loss_args, device=self.device)
        with torch.no_grad():
            z = self(x, *loss_args, output_layer=output_layer)[0]
            return get_np(self._loss_fn(z, x, *loss_args))

    def compute_loss(self, x, *loss_args):
        """Compute the loss  along the network's layers

        Parameters
        ----------
        x : ndarray, shape (n_samples, n_dim)
            Data to approximate.
        D: ndarray, shape (n_dim, n_features)
            Design matrix for the problem.
        reg : float
            L2 regularization parameter
        """
        x, *loss_args = check_tensor(x, *loss_args, device=self.device)
        loss = []
        with torch.no_grad():
            for output_layer in range(self.n_layers):
                z = self(x, *loss_args, output_layer=output_layer + 1)[0]
                loss.append(get_np(self._loss_fn(z, x, *loss_args)))
        return np.array(loss)

    def get_grad_x(self, x, *loss_args, output_layer=None,
                   return_loss=False, computation=None):
        """Compute the gradient of LogReg relative to beta with autodiff."""
        if computation is None:
            computation = self.gradient_computation

        x, *loss_args = check_tensor(x, *loss_args, device=self.device)

        if computation == 'autodiff':
            x = check_tensor(x, device=self.device, requires_grad=True)
        z = self(x, *loss_args, output_layer=output_layer)[0]
        res = self._get_grad_x(z, x, *loss_args, return_loss=return_loss,
                               computation=computation)

        if return_loss:
            return get_np(res[0]), get_np(res[1])
        return get_np(res)

    def _get_jabobian_zx(self, z, x, retain_graph=False):
        """Compute the Jacobian of z in x.
        """
        n_samples, n_features = z.shape
        assert n_samples % n_features == 0
        select = torch.eye(n_features, device=self.device)
        select = select.repeat_interleave(n_samples // n_features, 0)
        J = torch.autograd.grad(z, x, grad_outputs=select,
                                retain_graph=retain_graph)[0]
        assert J.shape == (n_samples, x.shape[1])
        return J.view(n_samples // n_features, n_features, -1)

    def transform_with_jacobian(self, x, *loss_args, output_layer=None,
                                log_iters=None, log_callbacks=None,
                                requires_grad=True):
        """Compute the Jacobian of z relative to x.
        """
        n_samples, n_dim = x.shape
        _, n_features = self._get_z0(x, *loss_args).shape

        x = check_tensor(x, device=self.device, requires_grad=True)
        *loss_args, = check_tensor(*loss_args, device=self.device)

        # Contruct the matrix to probe the jacobian
        x = x.repeat(n_features, 1)

        # change jacobian to only select outputs with the first n_samples
        assert log_callbacks is None
        log_callbacks = ['J']

        z, log = self(x, *loss_args, output_layer=output_layer,
                      log_iters=log_iters, log_callbacks=log_callbacks)
        J = get_np(self._get_jabobian_zx(z, x))
        return get_np(z)[:n_samples], J, log

    def transform(self, x, *loss_args, output_layer=None, log_iters=None,
                  log_callbacks=DEFAULT_CALLBACKS, requires_grad=False):
        """Compute the best z for the logreg with penalty reg.
        """
        # Compat numpy
        x, *loss_args = check_tensor(x, *loss_args, device=self.device)
        x = check_tensor(x, requires_grad=True)

        with nullcontext() if requires_grad else torch.no_grad():
            z, log = self(x, *loss_args, output_layer=output_layer,
                          log_iters=log_iters, log_callbacks=log_callbacks)

        return get_np(z), log
