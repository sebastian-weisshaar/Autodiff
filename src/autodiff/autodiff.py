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
    def __init__(self, function, input_parameters: list, seed: list):
        self.f = function
        self.input_parameters = []
        self.p_dim = len(input_parameters)
        if self.p_dim != len(seed):
            raise IndexError("Input parameters must be same length as seed")
        for i in range(self.p_dim):
            node = Node((i + 1) - self.p_dim, input_parameters[i], for_deriv=seed[i])
            self.input_parameters.append(node)
        self.seed = seed
        self.output_nodes = None
        self.f_dim = None

    def forward(self):
        """
        Computes derivative using forward mode AD

        :return: derivative value
        """
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

    def function_value(self):
        """
        Evaluates function at x (input parameters)

        :return: function value
        """
        if self.output_nodes is None:
            self.forward()
        if self.f_dim == 1:
            return self.output_nodes.value
        else:
            return [output_node.value for output_node in self.output_nodes]


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
        if isinstance(value, (float, int)):
            self.value = value
        else:
            raise TypeError("Node value must be an integer or float")
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
            new_name = self.__new_name__(other)
            value = self.value + other.value
            for_deriv = self.for_deriv + other.for_deriv
            back_deriv = {self.name: 1, other.name: 1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.__new_name__(other)
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
            new_name = self.__new_name__(other)
            value = self.value - other.value
            for_deriv = self.for_deriv - other.for_deriv
            back_deriv = {self.name: 1, other.name: -1}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.__new_name__(other)
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
            new_name = self.__new_name__(other)
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
            new_name = self.__new_name__(other)
            value = self.value * other.value
            for_deriv = self.for_deriv * other.value + other.for_deriv * self.value
            back_deriv = {self.name: other.value, other.name: self.value}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.__new_name__(other)
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
            new_name = self.__new_name__(other)
            value = self.value / other.value
            for_deriv = (self.for_deriv * other.value - self.value * other.for_deriv) / (other.value * other.value)
            back_deriv = {self.name: 1 / other.value, other.name: -self.value / (other.value * other.value)}
            parents = [self, other]
            new_node = Node(new_name, value, for_deriv=for_deriv, back_deriv=back_deriv,
                            parents=parents)
            other.child.append(new_node)

        elif isinstance(other, (float, int)):
            new_name = self.__new_name__(other)
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
            new_name = self.__new_name__(other)
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

    # TODO: fix naming for multidimensional f, and for when more than 2 independent parameters
    def __new_name__(self, other=None):
        """
        Compute the name of the new node

        Parameters
        ----------
        other: Node, float, integer
            Other object involved in creation of new node

        Return
        ------
        new_name: int
            name for new node
        """
        if isinstance(other, Node):
            if self.child or other.child:
                list_child_names = []
                if self.child:
                    list_child_names += [child.name for child in self.child]
                if other.child:
                    list_child_names += [child.name for child in other.child]
                new_name = max(list_child_names) + 1
            else:
                new_name = max(self.name, other.name) + 1
        else:
            if self.child:
                new_name = max([child.name for child in self.child]) + 1
            else:
                new_name = self.name + 1
        return new_name
    
    def __eq__(self,other):
        if isinstance(other, Node):
            s1=self.name==other.name
            s2=self.value==other.value
            s3=self.parents==other.parents
            s4=self.child==other.child
            s5=self.for_deriv==other.for_deriv
            s6=self.back_deriv==other.back_deriv
            s7=self.adjoint==other.adjoint
            if all([s1,s2,s3,s4,s5,s6,s7]):
                return True
            return False
        raise TypeError("Please compare Node with Node")

    def __str__(self):
        str_output = f'Name: {self.name}'
        if self.value:
            str_output += f'\nValue: {self.value}'
        if self.child:
            str_output += f'\nChildren: {[child.name for child in self.child]}'
        if self.parents:
            str_output += f'\nParents: {[parent.name for parent in self.parents]}'
        return str_output
