import sys
import pytest
import numpy as np
sys.path.insert(1, '../autodiff_NARS')
from autodiff import AutoDiff
from autodiff import Node


class TestAutoDiff:
    """Test class for autodiff class"""

    def test_init(self):
        """Test initialization of AutoDiff class"""
        with pytest.raises(AssertionError):
            f = ["We", "We"]
            ad = AutoDiff(f)
        with pytest.raises(TypeError):
            f = 2.0
            ad = AutoDiff(f)

    def test_derivative(self):
        """Test forward and reverse mode derivative of AutoDiff class"""

        # 1d input, 1d output
        def f(x):
            return x ** 2 + 3 * x

        ad = AutoDiff(f)
        assert ad.f(1) == 4
        assert ad.df(1) == 5
        assert ad.df(1) == 5
        assert ad.df(1, method="backward") == 5
        assert ad.df(1, method="backward") == 5

        with pytest.raises(TypeError):
            ad = AutoDiff(f)
            ad.df(1, method="This is not forward or backward")

        # 1d input multi output
        def f1(x):
            return 2 * x

        def f2(x):
            return 3 * x

        ad = AutoDiff([f1, f2])
        input = 4
        answer_f = [8, 12]
        answer_df = [[2], [3]]
        ad_answer = ad.f(input)
        ad_answer_df = ad.df(input)

        ad_answer_df_bw = ad.df(input, method="backward")
        assert all([answer_f[i] == ad_answer[i] for i in range(2)])
        assert all([all(ad_answer_df[i][x] == answer_df[i][x] for x in range(1)) for i in range(2)])
        assert all([all(ad_answer_df_bw[i][x] == answer_df[i][x] for x in range(1)) for i in range(2)])

        # Multi d input, 1-d output
        def f(x):
            return x[0] * 2 + x[1] * 3

        input = [1, 2]
        ad = AutoDiff(f)
        ad_answer = ad.f(input)
        ad_answer_df = ad.df(input)
        ad_answer_df_bw = ad.df(input, method="backward")
        answer_f = 8
        answer_df = [[2, 3]]
        assert ad_answer == answer_f
        assert all([all(ad_answer_df[i][x] == answer_df[i][x] for x in range(2)) for i in range(1)])
        assert all([all(ad_answer_df_bw[i][x] == answer_df[i][x] for x in range(2)) for i in range(1)])

        # Multi d input multi d output
        def f1(x):
            return x[0] * 1 + x[1] / 4

        def f2(x):
            return x[0] ** 2 + x[1] + 1

        seed = [1, 0]

        input = [1, 4]
        ad = AutoDiff([f1, f2])
        ad_answer = ad.f(input)
        ad_answer_df = ad.df(input)
        ad_answer_df_bw = ad.df(input, method="backward")
        answer_f = [2, 6]
        answer_df = [[1, 0.25], [2, 1]]

        ad_seed = ad.df(input, method="backward", seed=seed)
        df_seed = [1, 2]
        assert all([ad_seed[i] == df_seed[i] for i in range(2)])
        assert all([ad_answer[i] == answer_f[i] for i in range(len(answer_f))])
        assert all([all(ad_answer_df[i][x] == answer_df[i][x] for x in range(2)) for i in range(2)])
        assert all([all(ad_answer_df_bw[i][x] == answer_df[i][x] for x in range(2)) for i in range(2)])

    def test_function_value(self):
        """Test function value of AutoDiff class"""

    def test_dunder_call(self):
        """Test dunder call of AutoDiff class"""
        def f1(x):
            return x[0] * 1 + x[1] / 4

        def f2(x):
            return x[0] ** 2 + x[1] + 1

        input = [1, 4]
        ad = AutoDiff([f1, f2])
        ad_answer, ad_answer_df = ad(input)
        ad_answer, ad_answer_df_bw = ad(input, method="backward")

        answer_f = [2, 6]
        answer_df = [[1, 0.25], [2, 1]]
        assert all([ad_answer[i] == answer_f[i] for i in range(len(answer_f))])
        assert all([all(ad_answer_df[i][x] == answer_df[i][x] for x in range(2)) for i in range(2)])
        assert all([all(ad_answer_df_bw[i][x] == answer_df[i][x] for x in range(2)) for i in range(2)])

        with pytest.raises(TypeError):
            ad_answer, ad_answer_df = ad(input, "This is not forward or backward")


class TestNode:
    """Test class for Node class."""

    def test_init(self):
        """Test initialization of Node class"""
        name = 1
        value = 3
        child = Node(300, 2)
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
            # Node(1.0, 3)
        with pytest.raises(TypeError):
            Node(1, ["Wrong value input for node"])

    def test_equality(self):
        """Test equality dunder method of Node class"""
        node1 = Node(2, 2)
        node2 = Node(2, 2)
        assert node1 == node2

        with pytest.raises(TypeError):
            node1 == "2"

    def test_addition(self):
        """Test addition dunder method of Node class"""
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
        """Test subtraction dunder method of Node class"""
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
        with pytest.raises(TypeError):
            '4' - node_1

    def test_multiplication(self):
        """Test multiplicatio dunder method of Node class"""
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

        node_5 = constant * node_1
        assert node_5.name != node_1.name
        assert node_5.value == node_1.value * constant
        assert node_5.for_deriv == constant * node_1.for_deriv
        assert node_5.back_deriv == {node_1.name: constant}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1 * '4'

    def test_division(self):
        """Test division dunder method of Node class"""
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
        assert node_4.back_deriv == {node_1.name: 1 / constant}
        assert node_4 in node_1.child
        assert node_4.parents == [node_1]

        constant = -19
        node_5 = constant / node_1
        assert node_5.name != node_1.name
        assert node_5.value == constant / node_1.value
        assert node_5.for_deriv == -constant / (node_1.value * node_1.value)
        assert node_5.back_deriv == {node_1.name: -constant / (node_1.value * node_1.value)}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1 / '4'
        with pytest.raises(TypeError):
            '4' / node_1

    def test_negative(self):
        """Test division dunder method of Node class"""
        name_1, name_2 = 3, 4
        value_1, value_2 = 2.0, 4.3
        for_deriv_1, for_deriv_2 = 2.1, 3.01
        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)

        node_3 = -node_1
        assert node_3.value == -1 * node_1.value
        assert node_3.name != node_1.name
        assert node_3.for_deriv == (-1 * node_1.for_deriv)
        assert node_3.back_deriv == {node_1.name: -1}
        assert node_3 in node_1.child
        assert node_3.parents == [node_1]

    def test_power(self):
        """Test division dunder method of Node class"""
        name_1, name_2 = 3, 4
        value_1, value_2 = 2.0, 4.3
        for_deriv_1, for_deriv_2 = 2.1, 3.01
        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)
        node_2 = Node(name_2, value_2, for_deriv=for_deriv_2)
        node_3 = node_1 ** node_2
        assert node_3.value == node_1.value ** node_2.value
        assert node_3.name != node_1.name or node_3.name != node_2.name
        assert node_3.for_deriv == node_2.value * node_1.value ** (node_2.value - 1) * node_1.for_deriv + \
               np.log(node_1.value) * node_1.value ** node_2.value * node_2.for_deriv
        assert node_3.back_deriv == {node_1.name: node_2.value * node_1.value ** (node_2.value - 1),
                                     node_2.name: np.log(node_1.value) * node_1.value ** node_2.value}
        assert node_3 in node_1.child
        assert node_3 in node_2.child
        assert node_3.parents == [node_1, node_2]

        node_1 = Node(name_1, value_1, for_deriv=for_deriv_1)

        constant = 19
        node_5 = constant ** node_1
        assert node_5.name != node_1.name
        assert node_5.value == constant ** node_1.value
        assert node_5.for_deriv == node_1.for_deriv * np.log(constant) * constant ** node_1.value
        assert node_5.back_deriv == {node_1.name: np.log(constant) * constant ** node_1.value}
        assert node_5 in node_1.child
        assert node_5.parents == [node_1]

        with pytest.raises(TypeError):
            node_1 ** '4'

    
