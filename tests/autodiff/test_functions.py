import sys
sys.path.insert(1, './src/autodiff')
from functions import sin,cos,tan
import numpy as np


class TestFunction:
    def test_sin():
        test_number=np.random.normal(0,100)
        assert(sin(test_number)==np.sin(test_number))
    def test_cos():
        test_number=np.random.normal(0,100)
        assert(cos(test_number)==np.cos(test_number))
    def test_tan():
        test_number=np.random.normal(0,100)
        assert(tan(test_number)==np.tan(test_number))
    
    

