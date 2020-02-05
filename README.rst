
DiffOpt: Differentiable Iterative Optimization
===============================================

A package implementing some iterative optimization algorithms so they are differentiable, using `pytorch`_


- **Classical iterative optimization algorithms**: Gradient Descent and Sinkhorn.

- **Stochastic algorithms**: Stochastic gradient descent.


Install
-------

The package can be installed using :code:`pip install .`


Benchmarks
----------

For all benchmarks, running the benchmark will first compute it and store the results in :code:`benchmarks/outputs`. Running it again with option :code:`--plot` will display the results.


**Gradient Estimation**: The performance for the estimation of the gradient with :math:`g_1`, :math:`g_2` and :math:`g_3` can be display for different loss functions with the following commands:

.. code-block:: bash

    $ python benchmarks/gradient_estimate.py --n-jobs 4
    $ python benchmarks/gradient_estimate.py --plot


**SGD properties**: The final noise level of the estimates :math:`g_1`, :math:`g_2` and :math:`g_3` obtained with SGD and a constant step size and the estimation performance with a deacreasing step size can be displayed with:

.. code-block:: bash

    $ python benchmarks/noise_level.py --n-jobs 4 --n-average 10
    $ python benchmarks/noise_level.py --plot

**Wasserstein Barycenters**:

.. code-block:: bash

    $ python benchmarks/wasserstein_barycenter.py --gpu 0 --eps .05 --n-samples 20 --n-alpha 1000 --n-outer 5000 --step-size .05
    $ python benchmarks/wasserstein_barycenter.py --plot


.. Links to different projects


.. _pytorch: https://pytorch.org/