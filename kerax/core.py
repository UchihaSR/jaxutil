"""Core JAX interface.

PyTreeNode and PyTreeFactory enable user-defined classes to be registered
as JAX primitive types, making them compatible with functions 
such as jit, grad, vmap, etc.
"""
import abc
import collections
import copy
import functools
import types

from jax import tree_util
from jax import numpy as np


class PyTreeNode(type):
    """A metaclass which registers a class as a JAX PyTree node.

    Classes should implement a __totuple__ and __fromtuple__ method. 
    """
    def __new__(self, clsname, bases, clsdict):
        cls = super(PyTreeNode, self).__new__(self, clsname, bases, clsdict)
        tree_util.register_pytree_node(cls,
                                       lambda x: (cls.__totuple__(x), None),
                                       lambda _, x: cls.__fromtuple__(x))
        return cls


def instancemethod(f):
    def __pytreeinstancemethod(*args, **kwargs):
        return f(*args, **kwargs)
    return __pytreeinstancemethod


def instanceproperty(f):
    return PyTreeProperty(f)


class PyTreeProperty(object):
    def __init__(self, f):
        self.value = f


class PyTreeFactory(object):
    """A factory which creates PyTreeNodes.

    The buld_node() method is used to generate object instances.
    Any methods with the @instancemethod (or @instanceproperty) decorator 
    will be attached to the generated instance.

    The __instancecall__ method is used to override the __call__ method
    on generated instances.

    All fields on the generated instances must be predefined in the
    __pytreefields__() method.
    """
    def __init__(self):
        super(PyTreeFactory, self).__init__()

        factory = collections.namedtuple('PyTreeNode'+self.__class__.__name__,
                                          self.__pytreefields__())
        module = self
        class Factory(factory):
            def __call__(self, *args, **kwargs):
                return module.__instancecall__(self, *args, **kwargs)

            def __getattr__(self, item):
                attr = getattr(module, item)
                if isinstance(attr, PyTreeProperty):
                    return attr.value(module, self)
                elif isinstance(attr, types.MethodType) and (attr.__name__ == '__pytreeinstancemethod'):
                    return functools.partial(attr, self)
                else:
                    raise ValueError('Unknown attribute:'+item)

            def clone(self):
                return copy.deepcopy(self)

        Factory.__name__ = 'PyTreeFactory'+self.__class__.__name__
        self._factory = Factory

    @abc.abstractmethod
    def __pytreefields__(self):
        """Return a list of (string) field names."""

    def build_node(self, *args, **kwargs):
        return self._factory(*args, **kwargs)
