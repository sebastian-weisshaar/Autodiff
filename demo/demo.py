from autodiff_NARS import autodiff as AD
from autodiff_NARS.functions import log, sin, sinh, exp
import numpy as np


print("SCENARIO 1: Single (callable) Function + Scalar Input")
#user specified/created function 
def f(x):
    return log(x) + sin(x)

input = 1 #note: input is a scalar and not placed in a list 

ad = AD.AutoDiff(f) #initiate Autodiff Class object

function_value = ad.f(x=input) #obtain function value at x=1
df_forward = ad.df(x=input) #derivative using forward mode (note: default is forward mode)
df_backward = ad.df(x=input, method = 'backward') #derivative using the backward mode

print(function_value)
print(df_forward)
print(df_backward)

print("\n")

print("SCENARIO 2: Single Function + Multivariate Input")
#user specified/created function 
def f(x):
    return log(x[0]) + sin(x[1]) + x[2]

input = [1, 3 , 5]  
ad = AD.AutoDiff(f) #initiate Autodiff Class object

function_value = ad.f(x=input) #obtain function value at x=[1,3,5]
dfdx1 = ad.df(x=input, seed = [1, 0, 0]) #partial derivative of f with respect to x1. The seed must be the same dimension as the input of f 
df_backward = ad.df(x=input, method = 'backward') #derivative using the backward mode

print(function_value)
print(dfdx1)
print(df_backward)

print("\n")

print("SCENARIO 3: Multivariate Function + Scalar Input")
#user specified/created function 
def f1(x):
    return log(x) + sin(x) + x

def f2(x):
    return sinh(x) * exp(x) - x

def f3(x):
    return x * sin(x)

input = 3 #note: input is a scalar and not placed in a list 
ad = AD.AutoDiff([f1, f2, f3]) #initiate Autodiff Class object

function_value = ad.f(x=input) #obtain function value at x=3
df_forward = ad.df(x=input) #derivative using forward mode (note: default is forward mode)
df_backward = ad.df(x=input, method = 'backward') #derivative using the backward mode

print(function_value)
print(df_forward)
print(df_backward)


print("\n")

print("SCENARIO 4: Multivariate Function + Multivariate Input")
#user specified/created function 
def f1(x):
    return log(x[0]) + sin(x[1]) + x[2]

def f2(x):
    return sinh(x[0]) * exp(x[1]) - x[2]

def f3(x):
    return x[0] * sin(x[1]) / x[2]

input = [1, 3 , 5] 
ad = AD.AutoDiff([f1, f2, f3]) #initiate Autodiff Class object

function_value = ad.f(x=input) #obtain function value at x=[1,3,5]
dfdx1 = ad.df(x=input, seed = [1, 0, 0]) #partial derivative of f with respect to x1. The seed must be the same dimension as the input of f 
df_backward = ad.df(x=input, method = 'backward') #derivative using the backward mode

print(function_value)
print(dfdx1)
print(df_backward)


print("\n")

print("Newton's Method Implementation 1: Scalar Function")

# function to find the root of
def f(x):
    return x**2 -3

ad = AD.AutoDiff(f) 

# Newtons method
def newton(x0, max_iter=10000,tol=1e-6):
    x=x0

    for i in range(max_iter):

        x -= ad.f(x)/ad.df(x)

        if np.abs(ad.f(x)) < tol: 
            return x

    return False

sol = newton(10)
print(sol)



print("\n")

print("Newton's Method Implementation 1: Scalar Function")

def f1(x):
    return (x[0]**2)*(x[1]**3) - x[0]*x[1]**3-1

def f2(x):
    return (x[0]**3) - x[0]*x[1]**3-4


ad = AD.AutoDiff([f1,f2])

def newton(x0, max_iter=10000,tol=1e-6):
    x=x0

    for i in range(max_iter):

        f, df = ad(x)

        x -= np.linalg.multi_dot([np.linalg.inv(df),f])

        if np.linalg.norm(ad.f(x)) < tol: 
            return x

    return False

sol = newton([1,1])
print(sol)

print("\n")