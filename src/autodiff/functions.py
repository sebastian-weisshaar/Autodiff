import numpy as np 
from autodiff import Node 

#Completed functions: sin,cos,tan,arcsin,arcos, arctan, sinh,cosh,tanh, sqrt, exp, log, sigmoid
#TODO: Docstrings, commented code


def sin(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = np.cos(node.value)*node.for_deriv
        back_deriv = {node.name: np.cos(node.value)}
        new_node = Node(new_name, np.sin(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node
    elif ((type(node) == int) | (type(node) == float)):
        return np.sin(node)
    else:
        raise TypeError   

def cos(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = -np.sin(node.value)*node.for_deriv
        back_deriv = {node.name: -np.sin(node.value)}
        new_node = Node(new_name, np.cos(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.cos(node)
    else:
        raise TypeError


def tan(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = (np.sin(node.value)/np.cos(node.value))*node.for_deriv
        back_deriv = {node.name: (np.sin(node.value)/np.cos(node.value))}
        new_node = Node(new_name, np.tan(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.tan(node)
    else:
        raise TypeError

def arcsin(node):
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

def arcos(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = (-1/np.sqrt(1-(node.value)**2))*node.for_deriv
        back_deriv = {node.name: -1/np.sqrt(1-(node.value)**2)}
        new_node = Node(new_name, np.arccos(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.arccos(node)
    else:
        raise TypeError


def artan(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = (1/(1+(node.value)**2))*node.for_deriv
        back_deriv = {node.name: 1/(1+(node.value)**2)}
        new_node = Node(new_name, np.arctan(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.arctan(node)
    else:
        raise TypeError

def sinh(node):
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

def cosh(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = np.sinh(node.value)*node.for_deriv
        back_deriv = {node.name: np.sinh(node.value)}
        new_node = Node(new_name, np.cosh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.cosh(node)
    else:
        raise TypeError

def tanh(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = ((1/np.cosh(node.value))**2)*node.for_deriv
        back_deriv = {node.name: ((1/np.cosh(node.value))**2)}
        new_node = Node(new_name, np.tanh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node  
    elif ((type(node) == int) | (type(node) == float)):
        return np.tanh(node)
    else:
        raise TypeError

def sqrt(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = ((1/2)*node.value**(-1/2))*node.for_deriv
        back_deriv = {node.name: ((1/2)*node.value**(-1/2))}
        new_node = Node(new_name, np.sqrt(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.sqrt(node)
    else:
        raise TypeError

def exp(node, base = np.e):
    if base <= 0:
        raise TypeError
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = (np.log(base)*(base**node.value))*node.for_deriv
        back_deriv = {node.name: (np.log(base)*(base**node.value))}
        new_node = Node(new_name, base**(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return base**(node)
    else:
        raise TypeError

def log(node, base = np.e):
    if base <= 0:
        raise TypeError
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = (1/(np.log(base)*node.value))*node.for_deriv
        back_deriv = {node.name: (1/(np.log(base)*node.value))}
        new_node = Node(new_name, np.log(node.value)/np.log(base), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return np.log(node)/np.log(base)
    else:
        raise TypeError

def sigmoid(node):
    if isinstance(node, Node):
        new_name = node.name + 1
        for_deriv = ((np.exp(-node.value))/((np.exp(-node.value)+1)**2))*node.for_deriv
        back_deriv = {node.name: ((np.exp(-node.value))/((np.exp(-node.value)+1)**2))}
        new_node = Node(new_name, 1/(1+np.exp(-node.value)), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
        node.child.append(new_node)
        return new_node 
    elif ((type(node) == int) | (type(node) == float)):
        return 1/(1+np.exp(-node))
    else:
        raise TypeError
