# jaxutil
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.com/justinjfu/jaxlib.svg?token=fqvmvQ41UapBjhTyM6GX&branch=master)](https://travis-ci.com/justinjfu/jaxlib)

A collection of tools for making JAX programming easier.

## kerax

Kerax implements neural network modules and optimizers. Kerax registers network and optimizer variables as JAX primitives, making them fully compatible with jit, grad, and vmap.

Networks can be created using standard combinator machinery, such as using Sequential, Dense, and activations.
```python
network_initializer = module.Sequential([
            module.Dense(32),
            module.Relu(),
            module.Dense(2),
])
        
network = network_initializer(random.PRNGKey(0), (-1, input_dim))

# Run a forward pass on a batch of 10 points.
result = network(random.uniform(10, input_dim))
```
Implementing custom modules requires inheriting from the module base class and implementing initialize and forward methods (see [here](https://github.com/justinjfu/jaxlib/blob/master/kerax/networks.py)). Automatic initialization is an upcoming feature.

Optimizers take as input a network and gradient. Once quirk is that since updates are functionally pure and states are immutable, the step function outputs both a new optimizer and a new network. Make sure to use the new optimizer on the next gradient update.
```python
optimizer = optimizers.Adam()(network)

grad = jax.grad(loss_fn)(network)
optimizer, network = optimizer.step(network, grad)
```

## distributions

TODO: Basic distributions and bijectors.

## rl

TODO: immutable replay buffers
