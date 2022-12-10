from autodiff_NARS import autodiff as AD
from autodiff_NARS.functions import log, sin, sinh, exp


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

