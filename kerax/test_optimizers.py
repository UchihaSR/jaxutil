import copy
import functools
import unittest

from kerax import optimizers
from kerax import module

from jax import numpy as np
from jax import random
from jax import tree_util
from jax import grad
from jax import jit


def reconstruct_loss(net, x):
    return np.mean(np.square(net(x) - x))


class AdamTest(unittest.TestCase):

    def test_learn_identity(self):
        rng = random.PRNGKey(0)
        Network = module.Sequential([
            module.Dense(16),
            module.Relu(),
            module.Dense(2),
        ])
        network = Network(rng, (-1, 2))

        Optimizer = optimizers.Adam(lr=1e-1)
        optimizer = Optimizer(network)

        x0 = np.array([[1.0, 1.0], [0.5, 1.0], [3.0, 0.5]])

        initial_loss = reconstruct_loss(network, x0)
        for _ in range(50):
            optimizer, network = optimizer.step(network, functools.partial(reconstruct_loss, x=x0))
            loss = reconstruct_loss(network, x0)
        final_loss = loss
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.01)

    def test_step(self):
        params = np.array([1.0, 1.5])
        grads = np.array([0.5, 0.5])

        optimizer = optimizers.Adam(lr=1.0)(params)

        new_state, new_params = optimizer.step_flattened(params, grads)
        self.assertTrue(np.allclose(new_params, np.array([0.0, 0.5])))

        optimizer = optimizers.Adam(lr=0.0)(params)
        new_state, new_params = optimizer.step_flattened(params, grads)
        self.assertTrue(np.allclose(new_params, np.array([1.0, 2.0])))


if __name__ == '__main__':
    unittest.main()
