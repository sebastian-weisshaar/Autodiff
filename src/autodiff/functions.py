import numpy as np 
from autodiff import Node 

#Completed functions: sin,cos,tan,arcsin,arcos, arctan, sinh,cosh,tanh, sqrt, exp, log, sigmoid
#TODO: Docstrings, commented code


def sin(var):
    """Sine Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the sine

    """
    
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = np.cos(var.value)*var.for_deriv
        back_deriv = {var.name: np.cos(var.value)}
        new_node = Node(new_name, np.sin(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node
    elif ((type(var) == int) | (type(var) == float)):
        return np.sin(var)
    else:
        raise TypeError   

def cos(var):
    """Cosine Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the cosine

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = -np.sin(var.value)*var.for_deriv
        back_deriv = {var.name: -np.sin(var.value)}
        new_node = Node(new_name, np.cos(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.cos(var)
    else:
        raise TypeError


def tan(var):
    """Tangent Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the tangent
    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = (np.sin(var.value)/np.cos(var.value))*var.for_deriv
        back_deriv = {var.name: (np.sin(var.value)/np.cos(var.value))}
        new_node = Node(new_name, np.tan(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.tan(var)
    else:
        raise TypeError

def arcsin(node):
    """Arcsin Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Arcsin

    """
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = (1/np.sqrt(1-(node.value)**2))*node.for_deriv
        back_deriv = {node.name: 1/np.sqrt(1-(node.value)**2)}
        new_node = Node(new_name, np.arcsin(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.arcsin(node)
    else:
        raise TypeError

def arcos(var):
    """Arcos Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Arcos

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = (-1/np.sqrt(1-(var.value)**2))*var.for_deriv
        back_deriv = {var.name: -1/np.sqrt(1-(var.value)**2)}
        new_node = Node(new_name, np.arccos(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.arccos(var)
    else:
        raise TypeError


def artan(var):
    """Arctan Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Arctan

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = (1/(1+(var.value)**2))*var.for_deriv
        back_deriv = {var.name: 1/(1+(var.value)**2)}
        new_node = Node(new_name, np.arctan(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.arctan(var)
    else:
        raise TypeError

def sinh(node):
    """Sinh Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Sinh

    """
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = np.cosh(node.value)*node.for_deriv
        back_deriv = {node.name: np.cosh(node.value)}
        new_node = Node(new_name, np.sinh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.sinh(node)
    else:
        raise TypeError

def cosh(var):
    """Cosh Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Cosh

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = np.sinh(var.value)*var.for_deriv
        back_deriv = {var.name: np.sinh(var.value)}
        new_node = Node(new_name, np.cosh(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.cosh(var)
    else:
        raise TypeError

def tanh(var):
    """Tanh Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Tanh

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = ((1/np.cosh(var.value))**2)*var.for_deriv
        back_deriv = {var.name: ((1/np.cosh(var.value))**2)}
        new_node = Node(new_name, np.tanh(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node  
    elif ((type(var) == int) | (type(var) == float)):
        return np.tanh(var)
    else:
        raise TypeError

def sqrt(var):
    """Square Root Operator

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Square Root

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = ((1/2)*var.value**(-1/2))*var.for_deriv
        back_deriv = {var.name: ((1/2)*var.value**(-1/2))}
        new_node = Node(new_name, np.sqrt(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.sqrt(var)
    else:
        raise TypeError

def exp(var, base = np.e):
    """Square Root Operator

    INPUT
    =======
    var: a real number or Node object
    base: the base of the exponential function (default base is e)
    
    RETURNS
    =======
    output: a real number or new node object after taking the exponential 

    """
    
    if base <= 0:
        raise TypeError
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = (np.log(base)*(base**var.value))*var.for_deriv
        back_deriv = {var.name: (np.log(base)*(base**var.value))}
        new_node = Node(new_name, base**(var.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return base**var
    else:
        raise TypeError

def log(var, base = np.e):
    """Logarithm Operator

    INPUT
    =======
    var: a real number or Node object
    base: the base of the Logarithm (default base is e)
    
    RETURNS
    =======
    output: a real number or new node object after taking the Logarithm

    """
    if base <= 0:
        raise TypeError
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = (1/(np.log(base)*var.value))*var.for_deriv
        back_deriv = {var.name: (1/(np.log(base)*var.value))}
        new_node = Node(new_name, np.log(var.value)/np.log(base), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return np.log(var)/np.log(base)
    else:
        raise TypeError

def sigmoid(var):
    """Sigmoid Operator (i.e 1/(1 + e^{-var}))

    INPUT
    =======
    var: a real number or Node object
    
    RETURNS
    =======
    output: a real number or new node object after taking the Sigmoid

    """
    if isinstance(var, Node):
        new_name = var.name + 1
        for_deriv = ((np.exp(-var.value))/((np.exp(-var.value)+1)**2))*var.for_deriv
        back_deriv = {var.name: ((np.exp(-var.value))/((np.exp(-var.value)+1)**2))}
        new_node = Node(new_name, 1/(1+np.exp(-var.value)), for_deriv=for_deriv, back_deriv=back_deriv, parents=[var])
        var.child.append(new_node)
        return new_node 
    elif ((type(var) == int) | (type(var) == float)):
        return 1/(1+np.exp(-var))
    else:
        raise TypeError
