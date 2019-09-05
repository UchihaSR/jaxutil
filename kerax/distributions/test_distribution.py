from kerax.distributions import distribution

import numpy as onp
import unittest

from jax import numpy as np
from jax import random


class DiagonalGaussianTest(unittest.TestCase):

  def test_sample_shape(self):
    gaussian = distribution.DiagonalGaussian(np.array([0,0]), np.array([1,1]))
    sample = gaussian.sample(random.PRNGKey(0))
    self.assertEqual(sample.shape, (2,))

  def test_logprob(self):
    mean = onp.array([0,0], dtype=np.float32)
    std = onp.array([1,1], dtype=np.float32)
    sample = onp.array([1.0,2.0], dtype=np.float32)

    jax_gaussian = distribution.DiagonalGaussian(mean, std)
    jax_logprob = jax_gaussian.log_prob(sample)
    self.assertEqual(jax_logprob, -4.3378773)

  def test_logprob_batch(self):
    mean = onp.array([0,0], dtype=np.float32)
    std = onp.array([1,1], dtype=np.float32)
    sample = onp.array([[1.0,2.0], [0.5, 0.6]], dtype=np.float32)

    jax_gaussian = distribution.DiagonalGaussian(mean, std)

    jax_logprob = jax_gaussian.log_prob(sample)
    answer = [-4.3378773, -2.142877]
    self.assertTrue(np.allclose(jax_logprob, np.array(answer)))


class BernoulliTest(unittest.TestCase):

  def test_sample_shape(self):
    gaussian = distribution.Bernoulli(np.array([0.5,0.2]))
    sample = gaussian.sample(random.PRNGKey(0))
    self.assertEqual(sample.shape, (2,))

  def test_logprob(self):
    sample = onp.array([1.0], dtype=np.float32)

    jax_gaussian = distribution.Bernoulli(0.6)
    jax_logprob = jax_gaussian.log_prob(sample)
    self.assertEqual(jax_logprob, 0.6)

  def test_logprob_batch(self):
    sample = onp.array([1.0, 0.0], dtype=np.float32)

    jax_gaussian = distribution.Bernoulli(0.6)

    jax_logprob = jax_gaussian.log_prob(sample)
    answer = [0.6, 0.4]
    self.assertTrue(np.allclose(jax_logprob, np.array(answer)))

if __name__ == '__main__':
  unittest.main()

