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
            module.Dense(2),
        ])
        network = Network(rng, (-1, 2))

        Optimizer = optimizers.Adam(lr=5e-1)
        optimizer = Optimizer(network)

        x0 = np.array([[1.0, 1.0], [5.0, 0.0], [0.0, 5.0]])

        initial_loss = reconstruct_loss(network, x0)
        for _ in range(50):
            optimizer, network = optimizer.step(network, functools.partial(reconstruct_loss, x=x0))
        loss = reconstruct_loss(network, x0)
        final_loss = loss
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.02)

    def test_step(self):
        params = np.array([1.0, 1.5])
        grads = np.array([0.5, 0.5])

        optimizer_def = optimizers.Adam(lr=1.0)
        optimizer = optimizer_def(params)

        new_params, new_state = optimizers.Adam.step_flattened(optimizer_def, optimizer, params, grads)
        self.assertTrue(np.allclose(new_params, np.array([0.0, 0.5])))

        optimizer_def = optimizers.Adam(lr=0.0)
        optimizer = optimizer_def(params)

        new_params, new_state = optimizers.Adam.step_flattened(optimizer_def, optimizer, params, grads)
        self.assertTrue(np.allclose(new_params, np.array([1.0, 1.5])))


if __name__ == '__main__':
    unittest.main()
