
import sys
sys.path.insert(1, './src/autodiff')
import numpy as np
import pytest
from autodiff import AutoDiff
from autodiff import Node


class TestAutoDiff:
    """
    Test class for autodiff class.

    ...
    Methods
    -------
    test_init
    test_forward
    test_function_value
    """

    def test_init(self):
        def f_test(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return x1 * x2 - x3 / x2 + 3 * x3

        seed = [0, 0, 1]
        input_parameters = [2, 4, 6]
        test_AD = AutoDiff(f_test, input_parameters, seed)

        assert test_AD.f == f_test
        assert test_AD.input_parameters[0] == Node(-2, 2)
        assert test_AD.input_parameters[1] == Node(-1, 4)
        assert test_AD.input_parameters[2] == Node(0, 6)
        assert test_AD.seed == seed
        assert test_AD.p_dim == 3
        assert test_AD.output_nodes is None
        assert test_AD.f_dim is None


    def test_forward(self):
        def f_test(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return x1 * x2 - x3 / x2 + 3 * x3

        seed = [0, 0, 1]
        input_parameters = [2, 4, 6]
        test_AD = AutoDiff(f_test, input_parameters, seed)

        assert test_AD.output_nodes.for_deriv == -1 / 4 + 3

    def test_function_value(self):
        def f_test(x):
            x1, x2, x3 = x[0], x[1], x[2]
            return x1 * x2 - x3 / x2 + 3 * x3

        seed = [0, 0, 1]
        input_parameters = [2, 4, 6]
        test_AD = AutoDiff(f_test, input_parameters, seed)
        assert test_AD.function_value() == f_test(input_parameters)


class TestNode:
    # TODO improve testing of name of new nodes in operations
    """
    Test class for Node class.

    ...
    Methods
    -------
    test_init
    test_addition
    test_subtraction
    test_division
    test_multiplication
    """

    def test_init(self):
        name = 1
        value = 3
        child = []
        parents = []
        for_deriv = 3
        back_deriv = {2: 3, 4: 6}
        node = Node(name, value, child, parents, for_deriv, back_deriv)
        assert node.name == name
        assert node.value == value
        assert node.child == child
        assert node.parents == parents
        assert node.for_deriv == for_deriv
        assert node.back_deriv == back_deriv
        assert node.adjoint == 0

        name = 2
        value = 3
        node = Node(name, value)
        assert node.name == name
        assert node.value == value
        assert node.parents == []
        assert node.child == []
        assert node.for_deriv == 1
        assert node.back_deriv == {}
        assert node.adjoint == 0

        with pytest.raises(TypeError):
            Node('name', 3)
            Node(1.0, 3)

    def test_addition(self):
        name_1, name_2 = 1, 2
        value_1, value_2 = 3.0, 4.0
        for_deriv_1, for_deriv_2 = 2.0, 3
        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)
        node_2 = Node(name_2, value_2, for_deriv=for_deriv_2)
        node_3 = node_1 + node_2
        assert node_3.value == node_1.value + node_2.value
        assert node_3.name != node_1.name or node_3.name != node_2.name
        assert node_3.for_deriv == node_1.for_deriv + node_2.for_deriv
        assert node_3.back_deriv == {node_1.name: 1, node_2.name: 1}
        assert node_3 in node_1.child
        assert node_3 in node_2.child
        assert node_3.parents == [node_1, node_2]

        constant = 10
        node_4 = node_1 + constant
        assert node_4.name != node_1.name
        assert node_4.value == node_1.value + constant
        assert node_4.for_deriv == node_1.for_deriv
        assert node_4.back_deriv == {node_1.name: 1}
        assert node_4 in node_1.child
        assert node_4.parents == [node_1]

        constant = 12.0
        node_5 = constant + node_1
        assert node_5.name != node_1.name
        assert node_5.value == node_1.value + constant
        assert node_5.for_deriv == node_1.for_deriv
        assert node_5.back_deriv == {node_1.name: 1}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1 + '3'

    def test_subtraction(self):
        name_1, name_2 = 1, 9
        value_1, value_2 = -3.0, 190
        for_deriv_1, for_deriv_2 = 2.0, 3
        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)
        node_2 = Node(name_2, value_2, for_deriv=for_deriv_2)
        node_3 = node_1 - node_2
        assert node_3.value == node_1.value - node_2.value
        assert node_3.name != node_1.name or node_3.name != node_2.name
        assert node_3.for_deriv == node_1.for_deriv - node_2.for_deriv
        assert node_3.back_deriv == {node_1.name: 1, node_2.name: -1}
        assert node_3 in node_1.child
        assert node_3 in node_2.child
        assert node_3.parents == [node_1, node_2]

        constant = 10
        node_4 = node_1 - constant
        assert node_4.name != node_1.name
        assert node_4.value == node_1.value - constant
        assert node_4.for_deriv == node_1.for_deriv
        assert node_4.back_deriv == {node_1.name: 1}
        assert node_4 in node_1.child
        assert node_4.parents == [node_1]

        constant = 12.0
        node_5 = constant - node_1
        assert node_5.name != node_1.name
        assert node_5.value == constant - node_1.value
        assert node_5.for_deriv == -node_1.for_deriv
        assert node_5.back_deriv == {node_1.name: -1}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1 - '4'

    def test_multiplication(self):
        name_1, name_2 = 1, 2
        value_1, value_2 = 3.0, 4.0
        for_deriv_1, for_deriv_2 = 2.0, 3
        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)
        node_2 = Node(name_2, value_2, for_deriv=for_deriv_2)
        node_3 = node_1 * node_2
        assert node_3.value == node_1.value * node_2.value
        assert node_3.name != node_1.name or node_3.name != node_2.name
        assert node_3.for_deriv == node_1.for_deriv * node_2.value + node_2.for_deriv * node_1.value
        assert node_3.back_deriv == {node_1.name: node_2.value, node_2.name: node_1.value}
        assert node_3 in node_1.child
        assert node_3 in node_2.child
        assert node_3.parents == [node_1, node_2]

        constant = 10
        node_4 = node_1 * constant
        assert node_4.name != node_1.name
        assert node_4.value == node_1.value * constant
        assert node_4.for_deriv == constant * node_1.for_deriv
        assert node_4.back_deriv == {node_1.name: constant}
        assert node_4 in node_1.child
        assert node_4.parents == [node_1]

        node_5 = constant*node_1
        assert node_5.name != node_1.name
        assert node_5.value == node_1.value * constant
        assert node_5.for_deriv == constant * node_1.for_deriv
        assert node_5.back_deriv == {node_1.name: constant}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1*'4'

    def test_division(self):
        name_1, name_2 = 3, 4
        value_1, value_2 = 2.0, 4.3
        for_deriv_1, for_deriv_2 = 2.1, 3.01
        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)
        node_2 = Node(name_2, value_2, for_deriv=for_deriv_2)
        node_3 = node_1 / node_2
        assert node_3.value == node_1.value / node_2.value
        assert node_3.name != node_1.name or node_3.name != node_2.name
        assert node_3.for_deriv == (node_1.for_deriv * node_2.value - node_1.value * node_2.for_deriv) / \
               (node_2.value * node_2.value)
        assert node_3.back_deriv == {node_1.name: 1 / node_2.value, node_2.name: -node_1.value /
                                                                                 (node_2.value * node_2.value)}
        assert node_3 in node_1.child
        assert node_3 in node_2.child
        assert node_3.parents == [node_1, node_2]

        constant = 10
        node_4 = node_1 / constant
        assert node_4.name != node_1.name
        assert node_4.value == node_1.value / constant
        assert node_4.for_deriv == node_1.for_deriv / constant
        assert node_4.back_deriv == {node_1.name: 1}
        assert node_4 in node_1.child
        assert node_4.parents == [node_1]

        constant = -19
        node_5 = constant/node_1
        assert node_5.name != node_1.name
        assert node_5.value == constant/node_1.value
        assert node_5.for_deriv == -constant / (node_1.value * node_1.value)
        assert node_5.back_deriv == {node_1.name: -constant / (node_1.value * node_1.value)}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1/'4'
