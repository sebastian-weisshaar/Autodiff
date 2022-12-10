import copy
import random
import numpy as np
from typing import Union, Callable


class AutoDiff:
    """
    Autodiff class, for computing automatic derivatives of functions

    ...

    Attributes
    ----------
    f : function
        f(x) - the function of interest
    input_parameters: list
        x - list of the values of the independent variable x
    seed: list
        seed vector

    Methods
    -------
    forward
        returns the derivative of f(x) in the seed direction, computed using forward mode
    function_value
        returns the function value of f(x) evaluated at the input parameters(x)
    """

    def __init__(self, functions: list):
        if isinstance(functions, Callable):
            self.f_dim = 1
            self.function = [functions]
        elif isinstance(functions, list):
            assert all([isinstance(f_i,Callable) for f_i in functions]), "Provide a list of functions to instantiate the AutoDiff class"
            assert len(functions) != 1, "One dimensional function should be provided as a callable object and not a " \
                                        "list "
            self.f_dim = len(functions)
            self.function = functions
        else:
            raise TypeError
        self.input_nodes = []
        self.output_nodes = []

    def f(self, x):
        """
        Evaluates function at x (input parameters)

        :return: function value
        """
        assert isinstance(x, (float, int, list, np.ndarray))
        if isinstance(x, (list, np.ndarray)):
            assert len(x) != 1, "Scalar input x must not be an array or list."
        f = np.array([f_i(x) for f_i in self.function])
        if self.f_dim == 1:
            return f[0]
        return f

    def _create_input_nodes(self, input_vector, seed=None):
        if seed is None:
            seed = np.ones(self.x_dim)
        input_nodes = []
        for i in range(self.x_dim):
            node = Node((i + 1) - self.x_dim, input_vector[i], for_deriv=seed[i])
            input_nodes.append(node)
        input_nodes = [copy.deepcopy(input_nodes) for _ in range(self.f_dim)]
        return input_nodes

    def df(self, x, method="forward", seed=None):
        assert isinstance(x, (float, int, list, np.ndarray))
        if isinstance(x, (float, int)):
            input_vector = [x]

        else:
            input_vector = x
        self.x_dim = len(input_vector)
        self.output_nodes = []

        if method == "forward":
            output = np.zeros((self.f_dim, self.x_dim))
            jac_seed_matrix = np.identity(self.x_dim)
            for i in range(self.x_dim):
                jac_seed = jac_seed_matrix[i, :]
                self.input_nodes = self._create_input_nodes(input_vector, jac_seed)
                output[:, i] = self._forward()
                self.output_nodes = []

        elif method == "backward":
            self.input_nodes = self._create_input_nodes(input_vector)
            output = self._backward()
        else:
            raise TypeError("Method supported is either 'forward' or 'backward'")

        if seed:
            assert len(seed) == self.x_dim, "The seed vector must be the same shape as the input x"
            output = output @ np.asarray(seed)

        if self.x_dim == 1 and self.f_dim == 1:
            return output.flatten()[0]
        return output

    def _forward(self):
        """
        Computes derivative using forward mode AD

        :return: derivative value
        """

        for function_i, input_node in zip(self.function, self.input_nodes):

            if self.x_dim == 1: # and self.f_dim > 1
                input_node = input_node[0]

            output_node = function_i(input_node)
            self.output_nodes.append(output_node)

        for_deriv = []

        for output_node in self.output_nodes:
            output_node.adjoint = 1
            for_deriv.append(output_node.for_deriv)
        for_deriv = np.array(for_deriv)
        if self.f_dim == 1:
            return for_deriv[0]
        return for_deriv

    def _backward(self):
        """
        Computes derivative using reverse mode AD
        :return: derivative value
        """

        def recur_update(node):
            if node.parents:
                for parent in node.parents:
                    parent.adjoint += node.adjoint * node.back_deriv[parent.name]
                    recur_update(parent)

        self._forward()

        for output_node in self.output_nodes:
            recur_update(output_node)

        adjoints = np.zeros((self.f_dim, self.x_dim))
        for i, input_node in zip(range(self.f_dim), self.input_nodes):
            adjoints[i] = np.array([input_parameter_xi.adjoint for input_parameter_xi in input_node])
        return adjoints

    def __call__(self, x, method='forward', seed=None):
        if method == 'forward' or method == 'backward':
            df = self.df(x, method=method, seed=seed)
        else:
            raise TypeError("Method supported is either 'forward' or 'backward'")
        return self.f(x), df


class Node:
    """
    Node data structure

    ...

    Attributes
    ----------
    name: int
    value: int, float
    child: list
    parents: list
    for_deriv: int, float
    back_deriv: dict
    """

    def __init__(self, name: int, value, child=None, parents=[],
                 for_deriv=1, back_deriv={}):
        if isinstance(name, int):
            self.name = name
        else:
            raise TypeError("Node name must be an integer")
            
        try:
            float(value)
        except:
            raise TypeError("Node value must be float or integer")
        self.value = value
        self.parents = parents
        if child:
            self.child = child
        else:
            self.child = []
        self.for_deriv = for_deriv
        self.back_deriv = back_deriv  # deriv of current node with respect to parents
        self.adjoint = 0

    def __add__(self, other):
        """
        sum of node with other

        Parameters
        ----------
        other: Node, float, int
            Object to which to add to node

        Returns
        -------
        new_node: Node
            New node resulting from the addition
        """
        if isinstance(other, Node):
            new_name = self._new_name()
            value = self.value + other.value
            for_deriv = self.for_deriv + other.for_deriv
            back_deriv = {self.name: 1, other.name: 1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self._new_name()
            value = self.value + other
            for_deriv = self.for_deriv
            back_deriv = {self.name: 1}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __radd__(self, other):
        """
        sum of node with other

        Parameters
        ----------
        other: Node, float, int
            Object to which to add to node

        Returns
        -------
        new_node: Node
            New node resulting from the addition
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        difference between node and other

        Parameters
        ----------
        other: Node, float, int
            Object to which to subtract from node

        Returns
        -------
        new_node: Node
            New node resulting from the subtraction
        """
        if isinstance(other, Node):
            new_name = self._new_name()
            value = self.value - other.value
            for_deriv = self.for_deriv - other.for_deriv
            back_deriv = {self.name: 1, other.name: -1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self._new_name()
            value = self.value - other
            for_deriv = self.for_deriv
            back_deriv = {self.name: 1}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __rsub__(self, other):
        """
        difference between other and node

        Parameters
        ----------
        other: Node, float, int
            Object from which to subtract the node

        Returns
        -------
        new_node: Node
            New node resulting from the subtraction
        """

        if isinstance(other, (float, int)):
            new_name = self._new_name()
            value = other - self.value
            for_deriv = - self.for_deriv
            back_deriv = {self.name: -1}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __mul__(self, other):
        """
        multiplication between node and other

        Parameters
        ----------
        other: Node, float, int
            Object to multiply node by

        Returns
        -------
        new_node: Node
            New node resulting from the multiplication
        """
        if isinstance(other, Node):
            new_name = self._new_name()
            value = self.value * other.value
            for_deriv = self.for_deriv * other.value + other.for_deriv * self.value
            back_deriv = {self.name: other.value, other.name: self.value}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self._new_name()
            value = self.value * other
            for_deriv = self.for_deriv * other
            back_deriv = {self.name: other}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __rmul__(self, other):
        """
        multiplication between other and node

        Parameters
        ----------
        other: Node, float, int
            Object to multiply node by

        Returns
        -------
        new_node: Node
            New node resulting from the multiplication
        """
        return self.__mul__(other)

    def __pow__(self, other):
        """
        power between node and other

        Parameters
        ----------
        other: Node, float, int
            Object to multiply node by

        Returns
        -------
        new_node: Node
            New node resulting from the multiplication
        """
        if isinstance(other, Node):
            new_name = self._new_name()
            value = self.value ** other.value
            for_deriv = other.value*self.value**(other.value-1)*self.for_deriv + \
                        np.log(self.value)*self.value**other.value*other.for_deriv
            back_deriv = {self.name: other.value*self.value ** (other.value-1),
                          other.name: np.log(self.value)*self.value**other.value}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self._new_name()
            value = self.value ** other
            for_deriv = other*self.value ** (other-1)*self.for_deriv
            back_deriv = {self.name: other*self.value ** (other-1)}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __rpow__(self, other):
        new_name = self._new_name()
        value = other ** self.value
        for_deriv = self.for_deriv * np.log(other) * other ** self.value
        back_deriv = {self.name: np.log(other) * other ** self.value}
        parents = [self]
        new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                        parents=parents)
        self.child.append(new_node)
        return new_node

    def __truediv__(self, other):
        """
        division between other and node

        Parameters
        ----------
        other: Node, float, int
            Object to divide node by

        Returns
        -------
        new_node: Node
            New node resulting from the division
        """
        if isinstance(other, Node):
            new_name = self._new_name()
            value = self.value / other.value
            for_deriv = (self.for_deriv * other.value - self.value * other.for_deriv) / (other.value * other.value)
            back_deriv = {self.name: 1 / other.value, other.name: -self.value / (other.value * other.value)}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self._new_name()
            value = self.value / other
            for_deriv = self.for_deriv / other
            back_deriv = {self.name: 1/other}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __rtruediv__(self, other):
        """
        division between node and other

        Parameters
        ----------
        other: Node, float, int
            Object which is divided by node

        Returns
        -------
        new_node: Node
            New node resulting from the division
        """
        if isinstance(other, (float, int)):
            new_name = self._new_name()
            value = other / self.value
            for_deriv = -other / (self.value * self.value)
            back_deriv = {self.name: -other / (self.value * self.value)}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def _new_name(self):
        """
        Compute the name of the new node
        Return
        ------
        new_name: int
            name for new node
        """
        return random.getrandbits(64)

    def __eq__(self, other):
        """
        checks equality between node and other

        Parameters
        ----------
        other: Node, float, int
            Object which is checked to be equal to node

        Returns
        -------
        Boolean:
            True if node is equal to other, else false
        """
        if isinstance(other, Node):
            s1 = self.name == other.name
            s2 = self.value == other.value
            s3 = self.parents == other.parents
            s4 = self.child == other.child
            s5 = self.for_deriv == other.for_deriv
            s6 = self.back_deriv == other.back_deriv
            s7 = self.adjoint == other.adjoint
            if all([s1, s2, s3, s4, s5, s6, s7]):
                return True
            return False
        raise TypeError("Please compare Node with Node")

    def __neg__(self):
        """
        negation of node

        Returns
        -------
        new_node: Node
            New node resulting from negation
        """
        new_name = self._new_name()
        value = - self.value
        for_deriv = - self.for_deriv
        back_deriv = {self.name: -1}
        parents = [self]
        new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                        parents=parents)
        self.child.append(new_node)
        return new_node
