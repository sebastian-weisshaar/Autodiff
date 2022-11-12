import pytest


def f(x):
    return x**2

def test_1():
    assert f(2)==4