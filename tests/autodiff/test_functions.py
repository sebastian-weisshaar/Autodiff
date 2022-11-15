import sys
sys.path.insert(1, '/opt/github-runner/code1/_work/team19/team19/src/autodiff')
from functions import sin
import numpy as np

def test_sin():
    assert(sin(12)==np.sin(12))
