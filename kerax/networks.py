from jax import numpy as np

from kerax import module
from kerax import core


class Feedforward(module.Module):
    def __init__(self,
                 output_dim,
                 layers=tuple(),
                 activation=module.Relu,
                 **dense_kwargs):
        super(Feedforward, self).__init__()
        _layers = []
        for layer_size in layers:
            _layers.append(module.Dense(out_dim=layer_size, **dense_kwargs))
            _layers.append(activation())
        _layers.append(module.Dense(out_dim=output_dim))
        self.network = module.Sequential(_layers)

    def __pytreefields__(self):
        return ['network']

    def initialize(self, rng_key, input_shape):
        shape, net_params = self.network.initialize(rng_key, input_shape)
        return shape, self.build_node(network=net_params)

    @core.instancemethod
    def forward(self, params, inputs):
        return self.network.forward(params.network, inputs)


class FlattenNet(module.Module):
    """A network which concatenates inputs before feeding into a network."""

    def __init__(self, network_def):
        super(FlattenNet, self).__init__()
        self.network_def = network_def

    def __pytreefields__(self):
        return ['network']

    def initialize(self, rng_key, input_shape):
        shape, net_params = self.network_def.initialize(rng_key, input_shape)
        return shape, self.build_node(network=net_params)

    @core.instancemethod
    def forward(self, params, *inputs, **kwargs):
        flat_inputs = np.concatenate(inputs, axis=-1)
        return params.network.forward(flat_inputs, **kwargs)

