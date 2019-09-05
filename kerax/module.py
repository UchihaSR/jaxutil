"""Modules for JAX."""

import abc
import collections
import copy
import functools
from kerax import core

from jax import numpy as np
from jax import random
from jax.experimental import stax


class Module(core.PyTreeFactory):
    """Base module class.

    Subclasses must should define an initialize, __pytreefields___, and forward method.
    """

    def __init__(self):
        super(Module, self).__init__()

    def __call__(self, rng_key, input_shape):
        output_shape, params = self.initialize(rng_key, input_shape)
        return self.build_node(**params._asdict())

    def initialize(self, rng_key, input_shape):
        """Module initialization.

        Args:
            rng_key: JAX rng key.
            input_shape: Input tensor shape.

        Returns:
            output_shape: Output tensor shape
            params: Module parameters.
        """
        return input_shape, self.build_node()

    @abc.abstractmethod
    def forward(self, params, *args, **kwargs):
        """Module forward pass.

        Args:
            params: The module parameters.
        """
        pass

    def __instancecall__(self, params, *args, **kwargs):
        return self.forward(params, *args, **kwargs)


class Activation(Module):

    def __init__(self, activation_fn):
        super(Activation, self).__init__()
        self.activation_fn = activation_fn

    def __pytreefields__(self):
        return []

    def initialize(self, rng_key, input_shape):
        return input_shape, self.build_node()

    def forward(self, params, _input):
        return self.activation_fn(_input)


Relu = lambda: Activation(lambda x: np.maximum(x, 0))
Softplus = lambda: Activation(stax.softplus)
Tanh = lambda: Activation(np.tanh)
Sigmoid = lambda: Activation(stax.sigmoid)


class Dense(Module):

    def __init__(self, out_dim, W_init=stax.glorot(), b_init=stax.randn()):
        super(Dense, self).__init__()
        self.out_dim = out_dim
        self.W_init = W_init
        self.b_init = b_init

    def __pytreefields__(self):
        return ['W', 'b']

    def initialize(self, rng_key, input_shape):
        k1, k2 = random.split(rng_key)
        W, b = self.W_init(k1, (input_shape[-1], self.out_dim)), self.b_init(k2, (self.out_dim,))
        output_shape = input_shape[:-1] + (self.out_dim,)
        return output_shape, self.build_node(W=W, b=b)

    def forward(self, params, inputs):
        W, b = params.W, params.b
        return np.dot(inputs, W) + b


class Sequential(Module):

    def __init__(self, layers):
        super(Sequential, self).__init__()
        self.layers = layers

    def __pytreefields__(self):
        return ['layers']

    def initialize(self, rng_key, input_shape):
        params = []
        for layer in self.layers:
            rng_key, layer_rng = random.split(rng_key)
            input_shape, param = layer.initialize(layer_rng, input_shape)
            params.append(param)
        return input_shape, self.build_node(layers=params)

    def forward(self, params, inputs):
        for fun, param in zip(self.layers, params.layers):
            inputs = fun.forward(param, inputs)
        return inputs

