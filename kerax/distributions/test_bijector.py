from kerax.distributions import distribution
from kerax.distributions import bijector

import numpy as onp
import unittest

from jax import numpy as np
from jax import random


class TanhBijectorTest(unittest.TestCase):

  def test_sample_shape(self):
    gaussian = distribution.DiagonalGaussian(np.array([0,0]), np.array([1,1]))
    jax_bijector = bijector.TanhBijector(gaussian)
    sample = jax_bijector.sample(random.PRNGKey(0))
    self.assertEqual(sample.shape, (2,))

  def test_logprob_batch(self):
    mean = onp.array([0,0.2], dtype=np.float32)
    std = onp.array([1,1], dtype=np.float32)
    sample = onp.array([[0.1,-0.3], [0.5, 0.6]], dtype=np.float32)

    gaussian = distribution.DiagonalGaussian(mean, std)
    jax_bijector = bijector.TanhBijector(gaussian)

    jax_logprob = jax_bijector.log_prob(sample)
    answer = [-1.8683547, -1.3763735]
    self.assertTrue(np.allclose(jax_logprob, answer))


class ScaleBijectorTest(unittest.TestCase):

  def test_sample_shape(self):
    gaussian = distribution.DiagonalGaussian(np.array([0,0]), np.array([1,1]))
    jax_bijector = bijector.AffineBijector(gaussian, np.array([2.0, 2.0]))
    sample = jax_bijector.sample(random.PRNGKey(0))
    self.assertEqual(sample.shape, (2,))

  def test_logprob(self):
    gaussian1 = distribution.DiagonalGaussian(np.array([0,0]), np.array([1,1]))
    gaussian2 = distribution.DiagonalGaussian(np.array([1,0]), np.array([2,2]))
    jax_bijector = bijector.AffineBijector(gaussian1, scale=np.array([2,2]), translation=np.array([1,0]))
    sample = np.array([1.0, 2.0])

    bijector_logprob = jax_bijector.log_prob(sample)
    direct_logprob = gaussian2.log_prob(sample)

    self.assertTrue(np.allclose(bijector_logprob, direct_logprob))


class ChainBijectorTest(unittest.TestCase):

  def test_logprob(self):
    gaussian1 = distribution.DiagonalGaussian(np.array([0,0]), np.array([1,1]))
    gaussian2 = distribution.DiagonalGaussian(np.array([2,4]), np.array([4,2]))

    jax_bijector = bijector.chain(
      bijector.AffineBijector.chain(scale=np.array([2,2]),
                                                 translation=np.array([1,2])),
      bijector.AffineBijector.chain(scale=np.array([2,1]),
                                                 translation=np.array([0,2])),
    )(gaussian1)
    sample = np.array([1.0, 2.0])

    bijector_logprob = jax_bijector.log_prob(sample)
    direct_logprob = gaussian2.log_prob(sample)

    self.assertTrue(np.allclose(bijector_logprob, direct_logprob))


if __name__ == '__main__':
  unittest.main()

