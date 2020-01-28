# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt

from diffopt.sinkhorn import Sinkhorn


EPS = 1e-12


fontsize = 15
params = {"pdf.fonttype": 42,
          'axes.labelsize': fontsize,
          'font.size': fontsize,
          'legend.fontsize': fontsize + 2,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'text.usetex': True}
plt.rcParams.update(params)


def plot_gradients(n_iters, g1, g2, z_diff=None, title=None):

    eps = EPS
    g1, g2, z_diff = np.array(g1), np.array(g2), np.array(z_diff)

    plt.figure(figsize=(4, 3))
    plt.semilogy(n_iters, g1 + eps, label=r'$\|g_1^t - g^*\|$', linewidth=3)
    plt.semilogy(n_iters, g2 + eps, label=r'$\|g_2^t - g^*\|$', linewidth=3)
    if z_diff is not None:
        plt.semilogy(n_iters, np.array(z_diff) + eps, label=r'$\|z^t - z^*\|$', color='k',
                     linewidth=3, linestyle='dashed')
    x_ = plt.xlabel(r'$t$')
    y_ = plt.ylabel('')
    plt.legend()
    if title is not None:
        plt.savefig('figures/%s.pdf' % title, bbox_extra_artists=[x_, y_],
                    bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    n, p = 100, 30
    eps = .1
    rng = np.random.RandomState(42)
    a = rng.rand(n)
    b = rng.rand(p)
    a /= a.sum()
    b /= b.sum()
    ux = np.linspace(0, 1, n)
    uy = np.linspace(0, 1, p)
    C = (ux[:, None] - uy[None, :]) ** 2
    K = np.exp(-C / eps)

    max_iter = 5000
    n_iters = range(1, 40)

    sinkhorn = Sinkhorn(n_layers=max_iter, log_domain=False)

    # Compute true minimizer
    g_star = sinkhorn.gradient_beta_analytic(a, b, C, eps, max_iter)
    f_, g_ = sinkhorn.transform(a, b, C, eps)
    g1_list = []
    g2_list = []
    z_diff = []
    for n_iter in n_iters:
        # f, g = sinkhorn_np(a, b, C, eps, n_iter)
        f, g = sinkhorn.transform(a, b, C, eps, output_layer=n_iter)
        diff = np.sqrt(np.linalg.norm(f - f_) ** 2
                       + np.linalg.norm(g - g_) ** 2)
        z_diff.append(diff)
        g1 = sinkhorn.gradient_beta_analytic(a, b, C, eps, n_iter)
        g2 = sinkhorn.gradient_beta(a, b, C, eps, n_iter)
        g1_list.append(np.linalg.norm(g1 - g_star))
        g2_list.append(np.linalg.norm(g2 - g_star))
    plot_gradients(n_iters, g1_list, g2_list, z_diff)
