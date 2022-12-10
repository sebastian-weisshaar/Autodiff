
## workflow status
[![test](https://code.harvard.edu/CS107/team19/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team19/actions/workflows/test.yml)

[![coverage](https://code.harvard.edu/CS107/team19/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team19/actions/workflows/coverage.yml)


# autodiff-NARS

**An automatic differentiation package implementing forward and reverse mode**

Please refer to `docs/documentation.ipynb` for more details


| Group Members       | Emails                              |
|---------------------|-------------------------------------|
| Nora Hallqvist      | nhallqvisthellstadius@g.harvard.edu |
| Robin Robinson      | robin_robinson@college.harvard.edu        |
| Anna Midgley        | amidgley@g.harvard.edu              |
| Sebastian Weisshaar | sweisshaar@g.harvard.edu            |


# Installation 

To install from pip:

```Python
pip install autodiff-NARS
```
# Supported Elementary Operations

| Elementary Operation | Example |
|----------------------|---------|
| Addition             | x+y     |
| Multiplication       | x*y     |
| Division             | x/y     |
| Power                | x**2    |
| Subtraction          | x-y     |
| Negative             | -x      |

# Supported Unary Functions

| **Unary Function**  | **Example**       |
|---------------------|-------------------|
| sin                 | sin(x)            |
| cos                 | cos(x)            |
| tan                 | tan(x)            |
| arcsin              | arcsin(x)         |
| arccos              | arccos(x)         |
| arctan              | arctan(x)         |
| sinh                | sinh(x)           |
| cosh                | cosh(x)           |
| tanh                | tanh(x)           |
| sqrt                | sqrt(x)           |
| exp (any base)      | exp(x, base = np.e)  |
| log (any base)      | log(x, base = np.e)  |
| sigmoid             | sigmoid(x)        |


# Demo 

**Example 1:** 

Given the scalar function 
$$f(x) = \log(x) + \sin(x)$$

Below we demonstrate how to calculate the function value and derivative using both forward and backward mode at $x=1$.

```Python
import autodiff as AD
from functions import sin,log

def f(x):
    return log(x) + sin(x)

ad = AD.AutoDiff(f)
function_value = ad.f(x=1) 
df_forward = ad.df(x=1) 
df_backward = ad.df(x=1, method = 'backward') 

print(function_value)
> 0.8414709848078965
print(df_forward)
> 1.5403023058681398
print(df_backward)
> 1.5403023058681398
```

**Example 2:** 

Given the function: 

$$f(\boldsymbol{x})=\log(x_1) + \sin(x_2) + x_3$$

Below we demonstrate how to find the function value, Jacobian using reverse mode and partial derivative of $\frac{\partial f}{\partial x_1}$ using forward mode at point $x =[1,3,5]$ .

In order to obtain the partial derivative (or direction derivative), the user must specify a seed vector. The seed must be of the same dimensions as the input vector. 

```Python
import autodiff as AD
from functions import sin,log

def f(x):
    return log(x[0]) + sin(x[1]) + x[2]

input = [1, 3 , 5]  
ad = AD.AutoDiff(f) 
function_value = ad.f(x=input) 
dfdx1 = ad.df(x=input, seed = [1, 0, 0]) 
df_forward = ad.df(x=input) 
df_backward = ad.df(x=input, method = 'backward')


print(function_value)
> 5.141120008059867
print(dfdx1)
> [1.]
print(df_forward)
> [[ 1.        -0.9899925  1.       ]]
print(df_backward)
> [[ 1.        -0.9899925  1.       ]]
```

Above examples and including two more can be run by: 
```Python
from autodiff-NARS import demo
demo()
```