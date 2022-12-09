import sys
import numpy as np
sys.path.insert(1, './src/autodiff')
from autodiff import AutoDiff
from autodiff import Node

def f(x):
    return 2*x
ad=AutoDiff(f)
print(ad.f(3))
print(ad.df(3))
print(ad.df(3,"backward"))

def f2(x):
    return 3*x


print("Multi")
ad2=AutoDiff([f,f2])
print(ad2.f(3))
print(ad2.df(3))
print(ad2.df(3,"backward"))
