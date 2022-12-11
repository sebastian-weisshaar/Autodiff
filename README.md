
## workflow status
[![test](https://code.harvard.edu/CS107/team19/actions/workflows/test.yml/badge.svg)](https://code.harvard.edu/CS107/team19/actions/workflows/test.yml)

[![coverage](https://code.harvard.edu/CS107/team19/actions/workflows/coverage.yml/badge.svg)](https://code.harvard.edu/CS107/team19/actions/workflows/coverage.yml)


# autodiff-NARS

**An automatic differentiation package implementing forward and reverse mode**.

Please refer to the [documentation](https://code.harvard.edu/CS107/team19/blob/main/docs/documentation.ipynb) for more details.

| Group Members       | Emails                              |
|---------------------|-------------------------------------|
| Nora Hallqvist      | nhallqvisthellstadius@g.harvard.edu |
| Robin Robinson      | robin_robinson@college.harvard.edu  |
| Anna Midgley        | amidgley@g.harvard.edu              |
| Sebastian Weisshaar | sweisshaar@g.harvard.edu            |


# Installation 

To install from pip:

```Python
pip install -i https://test.pypi.org/simple/autodiff-NARS
```

# Uninstall 
How to uninstall:

```Python
pip uninstall autodiff-NARS
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

Given the scalar function: 
$$f(x) = \log(x) + \sin(x)$$

We demonstrate how to calculate the function value and derivative using both forward and backward mode at $x=1$.

```Python
from autodiff_NARS import autodiff as AD
from autodiff_NARS.functions import sin,log

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

We demonstrate how to find the function value, Jacobian using reverse mode, and the partial derivative of $\frac{\partial f}{\partial x_1}$ using forward mode at point $x =[1,3,5]$ .

In order to obtain the partial derivative (or direction derivative), the user must specify a seed vector. The seed vector must be of the same dimensions as the input vector. Specific to this example, to find the partial derivative with respect to $x_1$ the seed vector would be $[1,0,0].$


```Python
from autodiff_NARS import autodiff as AD
from autodiff_NARS.functions import sin,log

def f(x):
    return log(x[0]) + sin(x[1]) + x[2]

input = [1, 3 , 5]  
ad = AD.AutoDiff(f) 
function_value = ad.f(x=input) 
dfdx1 = ad.df(x=input, seed = [1, 0, 0]) 
df_backward = ad.df(x=input, method = 'backward')


print(function_value)
> 5.141120008059867
print(dfdx1)
> [1.]
print(df_backward)
> [[ 1.        -0.9899925  1.       ]]
```


**Example 3:** 

Given the multivariate function: 

$$f(\boldsymbol{x})=\begin{bmatrix}
\log(x_1) + \sin(x_2) + x_3\\ 
\sinh(x_1)* \exp(x_2) - x_3\\ 
x_1* \frac{\sin(x_2)}{x_3}
\end{bmatrix}$$

We demonstrate how to find the function value, Jacobian using reverse mode, and the partial derivative of $\frac{\partial f}{\partial x_1}$ using forward mode at point $x =[1,3,5]$ .

To create a multivariate function, the user must first define the individual functions and place them in a list. 

```Python
from autodiff_NARS import autodiff as AD
from autodiff_NARS.functions import sin,log, sinh, exp

def f1(x):
    return log(x[0]) + sin(x[1]) + x[2]

def f2(x):
    return sinh(x[0]) * exp(x[1]) - x[2]

def f3(x):
    return x[0] * sin(x[1]) / x[2]

input = [1, 3 , 5] 
ad = AD.AutoDiff([f1, f2, f3]) 
function_value = ad.f(x=input)
dfdx1 = ad.df(x=input, seed = [1, 0, 0]) 
df_backward = ad.df(x=input, method = 'backward') 


print(function_value)
> [ 5.14112001 18.60454697  0.028224  ]
print(dfdx1)
> [1.00000000e+00 3.09936031e+01 2.82240016e-02]
print(df_backward)
> [[ 1.00000000e+00 -9.89992497e-01  1.00000000e+00]
   [ 3.09936031e+01  2.36045470e+01 -1.00000000e+00]
   [ 2.82240016e-02 -1.97998499e-01 -5.64480032e-03]]
```


# Broader Impact & Inclusivity Statement

### **Broader Impact**
Our software enables a user to efficiently compute the automatic derivative of a complex function. There is a risk that this could be used by users to negatively impact society. For example one can use AD to find the optimal adversarial perturbation to a street sign to confuse self-driving vehicles, or to optimize energy efficiency of nuclear weapons. There are many other examples of misuse that would result in harm to human life. There are also examples of how the user may use the code to positively contribute to society. For example, training a neural network that leads to discovery of a beneficial drug, or optimizing a chemical process to reduce the amount of carbon emissions. We are uncertain of what users of our package will build with it. We will remain knowledgeable of who is using our code, and for what implications. If we see unethical actions, we will report them, and in extreme cases remove our package from PyPI. 

### **Inclusivity**
We as a team encourage and highly value code contribution from all coders, coming from diverse backgrounds. Examples of contributions that we value include translation of documentation into other languages, help with outreach and onboarding of new contributors, development of tutorials and other educational material. We do however acknowledge that various barriers exist for underrepresented groups in accessing the code base and contributing towards it. We are in the initial stages of the project, but plan to translate the documentation into other languages, and develop more in depth tutorials for how to use the code to improve accessibility.  Currently, pull requests are categorized and then assigned to the team member responsible for the area, for example testing. New pull requests are checked weekly, reviewed, and either approved or denied. If denied, the team member may provide recommended adjustments to the pull request. We plan to involve more people in this process as the project expands, to ensure that all possible contributions are considered. 
