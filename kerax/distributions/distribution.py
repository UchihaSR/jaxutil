import abc

from jax import random
from jax import numpy as np
from jax.experimental import stax


class Distribution(object):
    @abc.abstractmethod
    def sample(self, rng_key):
        """Draw a single sample from this distribution.

        For multiple samples, use vmap.
        """

    @abc.abstractmethod
    def log_prob(self, input):
        """Compute log probability of a point."""


class DiagonalGaussian(Distribution):
    logsq2pi = np.log(2*np.pi) * 0.5
    def __init__(self, mean, std):
        super(DiagonalGaussian, self).__init__()
        self.mean = mean
        self.std = std

    def sample(self, rng_key):
        return random.normal(rng_key, shape=self.mean.shape) * self.std + self.mean

    def log_prob(self, value):
        std = self.std
        logstd = np.log(self.std)
        gaussian_prob = - self.logsq2pi - logstd - \
            np.square(value - self.mean)/(2*np.square(std))
        return np.sum(gaussian_prob, axis=-1)


class Bernoulli(Distribution):

    def __init__(self, p):
        super(Bernoulli, self).__init__()
        self.p = p

    def sample(self, rng_key):
        return random.uniform(rng_key) <= self.p

    def log_prob(self, value):
        return value * self.p + (1-value) * (1-self.p)

