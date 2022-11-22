import numpy as np 
from autodiff import Node 

#Completed functions: sin,cos,tan,arcsin,arcos, arctan, sinh,cosh,tanh, sqrt, exp, log, sigmoid
#TODO: Docstrings, commented code


def sin(var):
    """
        Sine operator returns the sine of var object.

        Parameters
        ----------
        var: Node, float, int
            Object which to to apply the sine function

        Returns
        -------
        new_node: Node, int, float
            New node resulting from the applying the sine function  
        """
    
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = np.cos(var.value)*var.for_deriv
        back_deriv = {var.name: np.cos(var.value)}
        new_node = Node(new_name, np.sin(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node
    elif isinstance(var, (int, float)):
        return np.sin(var)
    else:
        raise TypeError   

def cos(var):
    """
        Cosine operator returns the cosine of var object.

        Parameters
        ----------
        var: Node, float, int
            Object which to to apply the cosine function

        Returns
        -------
        new_node: Node, int, float
            New node resulting from the applying the cosine function
        """
    if isinstance(var, Node):
        new_name =  var.__new_name__()
        for_deriv = -np.sin(var.value)*var.for_deriv
        back_deriv = {var.name: -np.sin(var.value)}
        new_node = Node(new_name, np.cos(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.cos(var)
    else:
        raise TypeError


def tan(var):
    """
        Tangent operator returns the tangent of var object.

        Parameters
        ----------
        var: Node, float, int
            Object which to to apply the tangent function

        Returns
        -------
        new_node: Node, float, int
            New node resulting from the applying the tangent function
        """
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = (1/np.cos(var.value)**2)*var.for_deriv
        back_deriv = {var.name: 1/np.cos(var.value)**2}
        new_node = Node(new_name, np.tan(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.tan(var)
    else:
        raise TypeError

def arcsin(var):
    """
        Arcsin operator returns the arcsin of var object.

        Parameters
        ----------
        var: Node, float, int
            Object which to to apply the arcsin function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the arcsin function
        """
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = (1/np.sqrt(1-(var.value)**2))*var.for_deriv
        back_deriv = {var.name: 1/np.sqrt(1-(var.value)**2)}
        new_node = Node(new_name, np.arcsin(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.arcsin(var)
    else:
        raise TypeError

def arccos(var):
    """
        Arcos operator returns the arcos of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the arcos function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the arcos function   
        """
    if isinstance(var, Node):
        new_name =  var.__new_name__()
        for_deriv = (-1/np.sqrt(1-(var.value)**2))*var.for_deriv
        back_deriv = {var.name: -1/np.sqrt(1-(var.value)**2)}
        new_node = Node(new_name, np.arccos(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.arccos(var)
    else:
        raise TypeError


def arctan(var):
    """
        Arctan operator returns the arctan of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the arctan function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the arctan function 
        """
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = (1/(1+(var.value)**2))*var.for_deriv
        back_deriv = {var.name: 1/(1+(var.value)**2)}
        new_node = Node(new_name, np.arctan(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.arctan(var)
    else:
        raise TypeError

def sinh(node):
    """
        Sinh operator returns the sinh of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the sinh function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the sinh function      
        """
    if isinstance(node, Node):
        new_name = var.__new_name__()
        for_deriv = np.cosh(node.value)*node.for_deriv
        back_deriv = {node.name: np.cosh(node.value)}
        new_node = Node(new_name, np.sinh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif isinstance(node, (int, float)):
        return np.sinh(node)
    else:
        raise TypeError

def cosh(var):
    """
        Cosh operator returns the cosh of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the cosh function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the cosh function      
        """
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = np.sinh(var.value)*var.for_deriv
        back_deriv = {var.name: np.sinh(var.value)}
        new_node = Node(new_name, np.cosh(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.cosh(var)
    else:
        raise TypeError

def tanh(var):
    """
        Tanh operator returns the tanh of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the tanh function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the tanh function      
        """
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = ((1/np.cosh(var.value))**2)*var.for_deriv
        back_deriv = {var.name: ((1/np.cosh(var.value))**2)}
        new_node = Node(new_name, np.tanh(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node  
    elif isinstance(var, (int, float)):
        return np.tanh(var)
    else:
        raise TypeError

def sqrt(var):
    """
        Sqaure Root operator returns the square root of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the sqaure root function

        Returns
        -------
        new_node: Node
            New node resulting from the applying the square root function      
        """
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = ((1/2)*var.value**(-1/2))*var.for_deriv
        back_deriv = {var.name: ((1/2)*var.value**(-1/2))}
        new_node = Node(new_name, np.sqrt(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.sqrt(var)
    else:
        raise TypeError

def exp(var, base = np.e):
    """
        Exponential operator returns the exponential of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the exponential function
        base: positive float, int
            The base of the exponential, default is set to e

        Returns
        -------
        new_node: Node
            New node resulting from the applying the exponential function
        """
    
    if base <= 0:
        raise TypeError
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = (np.log(base)*(base**var.value))*var.for_deriv
        back_deriv = {var.name: (np.log(base)*(base**var.value))}
        new_node = Node(new_name, base**(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return base**var
    else:
        raise TypeError

def log(var, base = np.e):
    """
        Logarithm operator returns the logarithm of var object.

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the logarithm function
        base: positive float, int
            The base of the logarithm, default is set to e

        Returns
        -------
        new_node: Node, float, int
            New node resulting from the applying the logarithm
        """
    if base <= 0:
        raise TypeError
    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = (1/(np.log(base)*var.value))*var.for_deriv
        back_deriv = {var.name: (1/(np.log(base)*var.value))}
        new_node = Node(new_name, np.log(var.value)/np.log(base), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return np.log(var)/np.log(base)
    else:
        raise TypeError

def sigmoid(var):
    """
        Sigmoid operator returns the sigmoid (i.e 1/(1 + e^{-var})) of var object 

        Parameters
        ----------
        var: Node, float, int
            Object to which to apply the sigmoid function

        Returns
        -------
        new_node: Node, float, int
            New node resulting from the applying the sigmoid function   
        """

    if isinstance(var, Node):
        new_name = var.__new_name__()
        for_deriv = ((np.exp(-var.value))/((np.exp(-var.value)+1)**2))*var.for_deriv
        back_deriv = {var.name: ((np.exp(-var.value))/((np.exp(-var.value)+1)**2))}
        new_node = Node(new_name, 1/(1+np.exp(-var.value)), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif isinstance(var, (int, float)):
        return 1/(1+np.exp(-var))
    else:
        raise TypeError
