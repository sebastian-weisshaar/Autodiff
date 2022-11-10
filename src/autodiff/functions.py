import numpy as np 
from autodiff import Node 

#Completed functions: sin,cos,tan,arcsin,arcos, arctan, sinh,cosh,tanh, 
def sin(node):
    new_name = node.name + 1
    for_deriv = np.cos(node.value)*node.for_deriv
    back_deriv = {node.name: np.cos(node.value)}
    new_node = Node(new_name, np.sin(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node

def cos(node):
    new_name = node.name + 1
    for_deriv = -np.sin(node.value)*node.for_deriv
    back_deriv = {node.name: -np.sin(node.value)}
    new_node = Node(new_name, np.cos(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def tan(node):
    new_name = node.name + 1
    for_deriv = (np.sin(node.value)/np.cos(node.value))*node.for_deriv
    back_deriv = {node.name: (np.sin(node.value)/np.cos(node.value))}
    new_node = Node(new_name, np.tan(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def arcsin(node):
    new_name = node.name + 1
    for_deriv = (1/np.sqrt(1-(node.value)**2))*node.for_deriv
    back_deriv = {node.name: 1/np.sqrt(1-(node.value)**2)}
    new_node = Node(new_name, np.arcsin(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def arcos(node):
    new_name = node.name + 1
    for_deriv = (-1/np.sqrt(1-(node.value)**2))*node.for_deriv
    back_deriv = {node.name: -1/np.sqrt(1-(node.value)**2)}
    new_node = Node(new_name, np.arccos(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def artan(node):
    new_name = node.name + 1
    for_deriv = (1/(1+(node.value)**2))*node.for_deriv
    back_deriv = {node.name: 1/(1+(node.value)**2)}
    new_node = Node(new_name, np.arctan(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def sinh(node):
    new_name = node.name + 1
    for_deriv = np.cosh(node.value)*node.for_deriv
    back_deriv = {node.name: np.cosh(node.value)}
    new_node = Node(new_name, np.sinh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def cosh(node):
    new_name = node.name + 1
    for_deriv = np.sinh(node.value)*node.for_deriv
    back_deriv = {node.name: np.sinh(node.value)}
    new_node = Node(new_name, np.cosh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node 

def tanh(node):
    new_name = node.name + 1
    for_deriv = ((1/np.cosh(node.value))**2)*node.for_deriv
    back_deriv = {node.name: ((1/np.cosh(node.value))**2)}
    new_node = Node(new_name, np.tanh(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node  

def sqrt(node):
    new_name = node.name + 1
    for_deriv = ((1/2)*node.value**(-1/2))*node.for_deriv
    back_deriv = {node.name: ((1/2)*node.value**(-1/2))}
    new_node = Node(new_name, np.sqrt(node.value), for_deriv=for_deriv, back_deriv=back_deriv, parents=[node])
    node.child.append(new_node)
    return new_node  