import abc
import functools

from jax import numpy as np
from jax.experimental import stax

from .distribution import Distribution


class Bijector(Distribution):
    def __init__(self, wrapped_distribution):
        super(Bijector, self).__init__()
        self.wrapped = wrapped_distribution

    @abc.abstractmethod
    def forward(self, value):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse(self, value):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_det_jacobian(self, value):
        raise NotImplementedError()

    def sample(self, rng_key):
        return self.forward(self.wrapped.sample(rng_key))

    def log_prob(self, value, inverse=None):
        if inverse is None:
            inverse = self.inverse(value)
        return -self.log_det_jacobian(inverse) + self.wrapped.log_prob(inverse)

    @classmethod
    def chain(cls, **kwargs):
        return functools.partial(cls, **kwargs)


def chain(*bijectors):
    def create(distribution):
        for bijector in bijectors:
            distribution = bijector(distribution)
        return distribution
    return create


class TanhBijector(Bijector):
    def __init__(self, wrapped):
        super(TanhBijector, self).__init__(wrapped)

    def forward(self, value):
        return np.tanh(value)

    def inverse(self, value):
        return np.arctanh(value)

    def log_det_jacobian(self, value):
        return np.sum(2. * (np.log(2.) - value - stax.softplus(-2. * value)), axis=-1)


class AffineBijector(Bijector):
    def __init__(self, wrapped, scale=1.0, translation=0.0):
        super(AffineBijector, self).__init__(wrapped)
        self.scale = scale
        self.translation = translation

    def forward(self, value):
        return self.scale * value + self.translation

    def inverse(self, value):
        return (1.0/self.scale) * (value - self.translation)

    def log_det_jacobian(self, value):
        return np.sum(np.log(self.scale), axis=-1)

