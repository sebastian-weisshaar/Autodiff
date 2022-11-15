# TODO: add docstrings to all classes and functions
class AutoDiff:
    def __init__(self, function, input_parameters: list, seed: list):
        self.f = function
        self.input_parameters = []
        self.p_dim = len(input_parameters)
        self.f_dim = None
        if self.p_dim != len(seed):
            raise IndexError("Input parameters must be same length as seed")
        for i in range(self.p_dim):
            node = Node((i + 1) - self.p_dim, input_parameters[i], for_deriv=seed[i])
            self.input_parameters.append(node)
        self.seed = seed
        self.output_nodes = None

    def forward(self):
        self.output_nodes = self.f(self.input_parameters)
        if isinstance(self.output_nodes, list):
            self.f_dim = len(self.output_nodes)
            for_deriv = []
            for output_node in self.output_nodes:
                output_node.adjoint = 1
                for_deriv.append(output_node.for_deriv)
            return for_deriv
        else:
            self.f_dim = 1
            self.output_nodes.adjoint = 1
            return self.output_nodes.for_deriv

# TODO: make backward able to handle multidimensional functions
    def backward(self):
        if self.output_nodes is None:
            self.forward()

        def recur_update(node):
            if node.parents:
                for parent in node.parents:
                    parent.adjoint += node.adjoint * node.back_deriv[parent.name]
                    return recur_update(parent)

        recur_update(self.output_nodes)

    def function_value(self):
        if self.output_nodes is None:
            self.forward()
        if self.f_dim == 1:
            return self.output_nodes.value
        else:
            return [output_node.value for output_node in self.output_nodes]


class Node:
    def __init__(self, name: int, value=None, child=[], parents=[],
                 for_deriv=1, back_deriv={}):
        self.name = name
        self.value = value
        self.child = child
        self.parents = parents
        self.for_deriv = for_deriv
        self.back_deriv = back_deriv  # deriv of current node with respect to parents
        self.adjoint = 0

    def __add__(self, other):
        if isinstance(other, Node):
            new_name = max(self.name, other.name) + 1
            value = self.value + other.value
            for_deriv = self.for_deriv + other.for_deriv
            back_deriv = {self.name: 1, other.name: 1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.name + 1
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
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Node):
            new_name = max(self.name, other.name) + 1
            value = self.value - other.value
            for_deriv = self.for_deriv - other.for_deriv
            back_deriv = {self.name: 1, other.name: -1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.name + 1
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
        if isinstance(other, Node):
            new_name = max(self.name, other.name) + 1
            value = other.value - self.value
            for_deriv = other.for_deriv - self.for_deriv
            back_deriv = {self.name: -1, other.name: 1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.name + 1
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
        if isinstance(other, Node):
            new_name = max(self.name, other.name) + 1
            value = self.value * other.value
            for_deriv = self.for_deriv * other.value + other.for_deriv * self.value
            back_deriv = {self.name: other.value, other.name: self.value}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.name + 1
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

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_name = max(self.name, other.name) + 1
            value = self.value / other.value
            for_deriv = (self.for_deriv * other.value - self.value * other.for_deriv) / (other.value * other.value)
            back_deriv = {self.name: 1 / other.value, other.name: -self.value / (other.value * other.value)}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.name + 1
            value = self.value / other
            for_deriv = self.for_deriv / other
            back_deriv = {self.name: 1}
            parents = [self]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
        else:
            raise TypeError
        self.child.append(new_node)
        return new_node

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, Node):
            new_name = max(self.name, other.name) + 1
            value = other.value / self.value
            for_deriv = (other.for_deriv * self.value - other.value * self.for_deriv) / (self.value * self.value)
            back_deriv = {self.name: -other.value / (self.value * self.value), other.name: 1 / self.value}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.name + 1
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


if __name__ == '__main__':
    def f_test(x):
        return [x[0] / x[1] + x[1] * x[0], x[1] * x[0]]


    seed = [1, 0]
    input_parameters = [1, 2]
    test_AD = AutoDiff(f_test, input_parameters, seed)
    print(test_AD.forward())
    # # test division
    # t1 = Node(1, 3.0, for_deriv=2)
    # t2 = Node(2, 4.0, for_deriv=3)
    # t3 = 3/t2
    # print(t3.value)
    # print(t3.name)
    # print(t3.for_deriv)
    # print(t3.back_deriv)

    # # test multiplication
    # t1 = Node(1, 3.0, for_deriv=2)
    # t2 = Node(2, 4.0, for_deriv=3)
    # t3 = 3*t2
    # print(t3.value)
    # print(t3.name)
    # print(t3.for_deriv)
    # print(t3.back_deriv)

    # test addition
    # t1 = Node(1, 3.0, for_deriv=2)
    # t2 = Node(2, 4.0, for_deriv=3)
    # t3 = t1 + t2
    # print(t3.value)
    # print(t3.name)
    # print(t3.for_deriv)
    # print(t3.back_deriv)
    # t3 = t1 + 10
    # print(t3.value)
    # print(t3.name)
    # print(t3.for_deriv)
    # print(t3.back_deriv)

    # test subtraction
    # t1 = Node(1, 3.0, for_deriv=2)
    # t2 = Node(2, 4.0, for_deriv=3)
    # t3 = t1 - t2
    # print(t3.value)
    # print(t3.name)
    # print(t3.for_deriv)
    # print(t3.back_deriv)
    # t3 = 10-t1
    # print(t3.value)
    # print(t3.name)
    # print(t3.for_deriv)
    # print(t3.back_deriv)
