import numpy as np
from ..utils import check_random_state


def make_ot(n_alpha=100, n_beta=30, point_dim=2, n_samples=1,
            random_state=None):
    """Generate an optimal transport problem

    Parameters
    ----------
    n_alpha, n_beta: int
        Dimension of the two input probabilities
    point_dim : int
        Dimension of the localisation of the two point clouds. The cost
        function will be chosen as the l2 distance between points chosen
        from random Gaussian(0, Id) in R^p, with p being point_dim.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.

    Return
    ------
    alpha : ndarray, shape (n_alpha,)
        First input distribution.
    beta: ndarray, shape (n_beta,)
        Second input distribution.
    C : ndarray, shape (n_alpha, n_beta)
        Cost matrix between the samples of each distribution.
    """

    rng = check_random_state(random_state)

    # Generate point cloud and probability distribution
    x = rng.randn(n_alpha, point_dim)
    y = rng.randn(n_beta, point_dim)
    alpha = abs(rng.rand(n_samples, n_alpha))
    beta = abs(rng.rand(n_beta))

    # normalize the probability
    alpha /= alpha.sum(axis=-1, keepdims=True)
    beta /= beta.sum()

    # Generate the cost matrix
    ux = np.linspace(0, 1, n_alpha)
    uy = np.linspace(0, 1, n_beta)
    C = (ux[:, None] - uy[None, :]) ** 2

    return alpha.squeeze(), beta, C, x, y
