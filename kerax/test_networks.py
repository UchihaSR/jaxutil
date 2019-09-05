from jax import numpy as np
from jax import random

from kerax import networks
from kerax import module
import unittest


class NetworksTest(unittest.TestCase):
    def test_sequential_shape_output(self):
        din = 3
        dout = 7
        network = module.Sequential([module.Dense(6), module.Softplus(), module.Dense(dout)])(random.PRNGKey(0), input_shape=(-1, din))
        output = network(np.ones([5, din]))
        self.assertEqual(output.shape, (5, dout))

    def test_feedforward_shape_output(self):
        din = 3
        dout = 5
        network = networks.Feedforward(dout, layers=(4,4,4))(random.PRNGKey(0), input_shape=(-1, din))
        output = network(np.ones([7, din]))
        self.assertEqual(output.shape, (7, dout))


if __name__ == '__main__':
    unittest.main()
