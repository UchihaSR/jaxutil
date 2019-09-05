from kerax import module
from kerax import optimizers
import unittest

from jax import numpy as np
from jax import random
from jax import tree_util

class StatelessModuleTest(unittest.TestCase):

    def test_dense_build(self):
        rng = random.PRNGKey(0)
        network = module.Dense(10)(rng, (-1, 5))

        result = network(np.zeros([2, 5]))
        self.assertEqual(result.shape, (2, 10))

    def test_dense_value(self):
        rng = random.PRNGKey(0)
        network = module.Dense(10)(rng, (-1, 5))

        result = network(np.zeros([2, 5]))
        self.assertTrue(np.allclose(result, network.b))

        result = network(np.ones([1, 5]))
        self.assertTrue(np.allclose(result[0], np.sum(network.W, axis=0) + network.b))

    def test_serial_build(self):
        rng = random.PRNGKey(0)
        Network = module.Sequential([
            module.Dense(10),
            module.Relu(),
            module.Dense(10),
        ])
        network = Network(rng, (-1, 5))

        result = network(np.zeros([2, 5]))
        self.assertEqual(result.shape, (2, 10))


if __name__ == '__main__':
    unittest.main()
