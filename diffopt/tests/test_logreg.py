import pytest
import autograd.numpy as np
from diffopt.logreg import LogReg
from autograd import grad, jacobian


def logreg_loss(x, D, z, lbda):
    res = - x * np.dot(D, z)
    return np.mean(np.log1p(np.exp(res))) + .5 * lbda * np.sum(z ** 2)


def gradient_descent(x, D, lbda, step, n_iter):
    n, p = D.shape
    z = np.zeros(p)
    for i in range(n_iter):
        grad_z = np.dot(D.T, - x / (1. + np.exp(x * np.dot(D, z)))) / n
        grad_z += lbda * z
        z -= step * grad_z
    return z


def gradient_descent_loss(x, D, lbda, step, n_iter):
    z = gradient_descent(x, D, lbda, step, n_iter)
    return logreg_loss(x, D, z, lbda)


def d2(x, D, z, lbda):
    n, p = D.shape
    u = np.dot(D, z)
    res = x * u
    f_res = np.exp(res) / (1 + np.exp(res)) ** 2
    dzz = np.dot(D.T, (x ** 2 * f_res)[:, None] * D) / n
    dzz += lbda * np.eye(p)
    dxz = D * (u * x * f_res - 1 / (1. + np.exp(res)))[:, None] / n
    return dzz, dxz


def grad_analytic(x, D, lbda, step, n_iter):
    n, p = D.shape
    z = gradient_descent(x, D, lbda, step, n_iter)
    return -np.dot(D, z) / (1. + np.exp(x * np.dot(D, z))) / n


def grad_implicit(x, D, lbda, step, n_iter):
    n, p = D.shape
    z = gradient_descent(x, D, lbda, step, n_iter)
    dzz, dxz = d2(x, D, z, lbda)
    dx = -np.dot(D, z) / (1. + np.exp(x * np.dot(D, z))) / n
    dz = np.dot(D.T, - x / (1. + np.exp(x * np.dot(D, z)))) / n
    dz += lbda * z
    return dx - np.dot(dxz, np.linalg.solve(dzz, dz))


grad_autodiff = grad(gradient_descent_loss)


@pytest.mark.parametrize('n_iter', [1, 10, 100, 1000])
def test_logreg_np(n_iter):

    n, p = 10, 30
    reg = 1.3
    rng = np.random.RandomState(0)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    logreg = LogReg(n_layers=n_iter)
    z_star, _ = logreg.transform(x, D, reg)

    step = 1 / (np.linalg.norm(D, ord=2) ** 2 / 4 / n + reg)
    print(np.linalg.norm(D, ord=2))

    z_np = gradient_descent(x.reshape(-1), D, reg, step, n_iter)
    assert np.allclose(z_np[None], z_star)

    loss = logreg.score(x, D, reg)
    loss_np = logreg_loss(x, D, z_np, reg)
    assert np.isclose(loss, loss_np)


def test_gradient_definition():

    n_iter = 1000
    n, p = 10, 30
    reg = 1.3
    rng = np.random.RandomState(0)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    step = 1 / (np.linalg.norm(D, ord=2) ** 2 / 4 / n + reg)
    g1 = grad_analytic(x.reshape(-1), D, reg, step, n_iter)
    g2 = grad_autodiff(x.reshape(-1), D, reg, step, n_iter)
    g3 = grad_implicit(x.reshape(-1), D, reg, step, n_iter)

    assert np.allclose(g2, g1)
    assert np.allclose(g2, g3)


@pytest.mark.parametrize('n_iter', [1, 10, 100, 1000])
@pytest.mark.parametrize('grad, f_grad', [('analytic', grad_analytic),
                                          ('implicit', grad_implicit),
                                          ('autodiff', grad_autodiff)])
def test_gradient(n_iter, grad, f_grad):

    n, p = 10, 30
    reg = 1.3
    rng = np.random.RandomState(0)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    step = 1 / (np.linalg.norm(D, ord=2) ** 2 / 4 / n + reg)
    g_np = f_grad(x.reshape(-1), D, reg, step, n_iter)

    # Compute gradient with default parameters
    logreg_ana = LogReg(n_layers=n_iter, gradient_computation=grad)
    g_star = logreg_ana.get_grad_x(x, D, reg)
    assert np.allclose(g_np[None], g_star)

    # Compute gradient changing the parameter
    with pytest.raises(NotImplementedError):
        g_star = logreg_ana.get_grad_x(x, D, reg, computation='fake')

    g_star = logreg_ana.get_grad_x(x, D, reg, computation=grad)
    assert np.allclose(g_np[None], g_star)


@pytest.mark.parametrize('n_iter', [1, 10, 100, 1000])
def test_jacobian(n_iter):

    n, p = 10, 30
    reg = 1.3
    rng = np.random.RandomState(0)
    D = rng.randn(n, p)
    x = rng.randn(1, n)

    # Compute true minimizer
    logreg_ana = LogReg(n_layers=n_iter, gradient_computation='autodiff')
    z_star, J_star, _ = logreg_ana.transform_with_jacobian(x, D, reg)

    step = 1 / (np.linalg.norm(D, ord=2) ** 2 / 4 / n + reg)
    auto_jacobian = jacobian(gradient_descent)
    J_np = auto_jacobian(x.reshape(-1), D, reg, step, n_iter)
    assert np.allclose(J_np[None], J_star)
