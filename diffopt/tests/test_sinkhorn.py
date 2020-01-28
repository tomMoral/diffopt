import pytest
import numpy as np
from adopty.sinkhorn import Sinkhorn
from adopty.datasets.optimal_transport import make_ot


@pytest.mark.parametrize(
    'eps', [.1, 1, 10]
)
@pytest.mark.parametrize(
    'n_layers', [1, 5, 10, 100, 1000]
)
def test_log_domain(eps, n_layers):
    """Test that the log domain computation is equivalent to classical sinkhorn
    """
    p = 2
    n, m = 10, 15

    alpha, beta, C, *_ = make_ot(n, m, p, random_state=0)
    alpha = np.r_['0,2', alpha, alpha]

    snet1 = Sinkhorn(n_layers, log_domain=True)
    f1, g1 = snet1.transform(alpha, beta, C, eps)
    snet2 = Sinkhorn(n_layers, log_domain=False)
    f2, g2 = snet2.transform(alpha, beta, C, eps)
    assert np.allclose(f1, f2)
    assert np.allclose(g1, g2)

    # Check that the scores are well computed
    assert np.isclose(
        snet1.score(alpha, beta, C, eps),
        snet2.score(alpha, beta, C, eps)
    )


def sinkhorn(a, b, K, n_iter):
    n, m = K.shape
    v = np.ones(m)
    for i in range(n_iter):
        u = a / np.dot(K, v)
        v = b / np.dot(u, K)
    return u, v

@pytest.mark.parametrize('n_samples', [2, 3, 5])
@pytest.mark.parametrize('log_domain', [True, False],
                         ids=lambda x: 'log_domain' if x is True else 'exp')
def test_sinkhorn_np(n_samples, log_domain):
    p = 2
    n, m = 10, 15
    eps = .1
    n_layers = 500

    alphas, beta, C, *_ = make_ot(n, m, p, n_samples=n_samples, random_state=0)

    snet = Sinkhorn(n_layers=n_layers, log_domain=log_domain)
    f, g = snet.transform(alphas, beta, C, eps)

    for i in range(n_samples):
        u, v = sinkhorn(alphas[i], beta, np.exp(-C / eps), n_layers)
        assert np.allclose(f[i], eps * np.log(u))
        assert np.allclose(g[i], eps * np.log(v))


@pytest.mark.parametrize('n_samples', [1, 2, 5])
@pytest.mark.parametrize('log_domain', [True, False],
                         ids=lambda x: 'log_domain' if x is True else 'exp')
@pytest.mark.parametrize('gradient', ['analytic', 'autodiff'])
def test_gradient_beta(n_samples, log_domain, gradient):
    p = 2
    n, m = 10, 15
    eps = 1
    n_layers = 100

    alphas, beta, C, *_ = make_ot(n, m, p, n_samples=n_samples, random_state=0)

    snet = Sinkhorn(n_layers=n_layers, log_domain=log_domain,
                    gradient_computation=gradient)
    snet_star = Sinkhorn(n_layers=1000, log_domain=False,
                         gradient_computation='analytic')

    f, g = snet.transform(alphas, beta, C, eps)
    f_star, g_star = snet_star.transform(alphas, beta, C, eps)
    err_norm = np.sqrt(np.linalg.norm((f - f_star).ravel()) ** 2 +
                       np.linalg.norm((g - g_star).ravel()) ** 2)
    assert err_norm < 1e-6

    # Get the gradient with analytic formula and autodiff
    G = snet.gradient_beta(alphas, beta, C, eps)
    G_star = snet_star.gradient_beta(alphas, beta, C, eps)

    assert np.allclose(G, G_star)
