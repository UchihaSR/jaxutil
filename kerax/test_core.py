import six

from kerax import core
import unittest

from jax import tree_util
from jax import numpy as np


@six.add_metaclass(core.PyTreeNode)
class DummyPyTreeNode(object):
    def __init__(self, a, b):
        self.a = np.array([a])
        self.b = np.array([b])

    def __totuple__(self):
        return (self.a, self.b)

    @classmethod
    def __fromtuple__(cls, values):
        return DummyPyTreeNode(values[0], values[1])


class DummyTupleFactory(core.PyTreeFactory):
    def __init__(self, a, b):
        super(DummyTupleFactory, self).__init__()
        self.a = a
        self.b = b

    def __pytreefields__(self):
        return ['a', 'b']

    def __call__(self):
        return self.build_node(a=np.array([self.a]),
                               b=np.array([self.b]))

    @core.instancemethod
    def get_sum(self, node):
        return node.a + node.b

    @core.instanceproperty
    def a_val(self, node):
        return node.a


class UtilTest(unittest.TestCase):

    def test_py_tree_node(self):
        test_input = [DummyPyTreeNode(7,2), DummyPyTreeNode(0,1)]

        test_output = tree_util.tree_map(lambda x: x+1, test_input)

        self.assertEqual(test_output[0].a, 8)
        self.assertEqual(test_output[0].b, 3)
        self.assertEqual(test_output[1].a, 1)
        self.assertEqual(test_output[1].b, 2)

    def test_py_tree_factory(self):
        factory = DummyTupleFactory(2, 5)

        instance = factory()
        test_output = tree_util.tree_map(lambda x: x+1, [instance])[0]

        self.assertEqual(test_output.get_sum(), 9.0)
        self.assertEqual(test_output.a_val, 3.0)


if __name__ == '__main__':
    unittest.main()
