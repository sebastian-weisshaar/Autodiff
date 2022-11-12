import pytest


def f(x):
    return x*3

def empty(y):
    return y+4

def test_1():
    assert f(2)==3