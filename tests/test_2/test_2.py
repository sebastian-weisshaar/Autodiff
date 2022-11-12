import pytest


def f(x):
    return x*3

def test_1():
    assert f(2)==4