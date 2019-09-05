import abc

import jax
from jax import flatten_util
from jax import numpy as np
from jax import lax
import numpy as onp

from kerax import core


class GradientOptimizer(core.PyTreeFactory):
    """Base class for gradient-based optimizers."""

    def __call__(self, params):
        param_flat, _ = flatten_util.ravel_pytree(params)
        return self.initialize(param_flat)

    @core.instancemethod
    def step(self, state, params, loss):
        """Perform one gradient descent update.

        Args:
            state: Optimizer state.
            params: Model parameters.
            loss: A scalar loss function which takes in a model
                parameters as a single argument.

        Returns:
            new_state: The updated optimizer state.
            new_params: The updated model parameters.
        """
        grads = jax.grad(loss)(params)
        param_flat, unflattener = flatten_util.ravel_pytree(params)
        grad_flat, _ = flatten_util.ravel_pytree(grads)
        new_param, new_state = self.step_flattened(state, param_flat, grad_flat)
        return new_state, unflattener(new_param)

    @abc.abstractmethod
    def step_flattened(self, state, params, grads):
        pass

    @abc.abstractmethod
    def initialize(self, params):
        pass

    @classmethod
    def __pytreefields__(cls):
        return []


class Adam(GradientOptimizer):
    """Adam optimizer

    Adam: A Method for Stochastic Optimization
    Diederik P. Kingma, Jimmy Ba
    https://arxiv.org/abs/1412.6980
    """

    def __init__(self, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
        """Adam optimizer.

        Args:
          lr: The step size or learning rate.
          b1: The exponential decay rate for the 1st moment estimates.
          b2: The exponential decay rate for the 2nd moment estimates.
          eps: A small constant for numerical stability.
        """
        super(Adam, self).__init__()
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    @classmethod
    def __pytreefields__(cls):
        return ['steps', 'm', 'v']

    def initialize(self, param):
        m0 = np.zeros_like(param)
        v0 = np.zeros_like(param)
        return self.build_node(steps=0, m=m0, v=v0)

    def step_flattened(self, state, param, grad):
        step, m, v = state
        m = (1 - self.b1) * grad + self.b1 * m  # First  moment estimate.
        v = (1 - self.b2) * (grad**2) + self.b2 * v  # Second moment estimate.
        mhat = m / (1 - self.b1**(step + 1))  # Bias correction.
        vhat = v / (1 - self.b2**(step + 1))
        param = param - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return param, self.build_node(steps=step + 1, m=m, v=v)


class GradientDescent(GradientOptimizer):
    def __init__(self, lr=5e-3):
        """Gradient descent optimizer.

        Args:
          lr: The step size or learning rate.
        """
        super(GradientDescent, self).__init__()
        self.lr = lr

    @classmethod
    def __pytreefields__(cls):
        return []

    def initialize(self, param):
        return self.build_node()

    def step_flattened(self, state, param, grad):
        param = param - self.lr * grad
        return param, self.build_node()


