import sys
sys.path.insert(1, './src/autodiff')
from functions import sin,cos,tan,arcsin,arcos,artan,sinh,cosh,tanh,sqrt,exp,log,sigmoid
from autodiff import Node
import numpy as np
import pytest

class TestFunction:
    
    def create_node(self,value_range=[-100,100]):
        p1,p2=value_range
        value=np.random.choice([np.random.uniform(p1,p2),np.random.randint(p1,p2)])
        for_deriv=np.random.choice([np.random.normal(12,10),np.random.randint(-200,300)])
        test_node=Node(np.random.randint(-1,1000),value,[],[],for_deriv,{})
        return test_node

    def helper_test(self,fun,numpy_equivalent,numpy_derivative,input_range=[-100,100],derivative_range=[-100,100]):

        test_node=self.create_node(value_range=input_range)
        new_node=fun(test_node)
        p1,p2=input_range
        test_number=np.random.choice([np.random.uniform(p1,p2),np.random.randint(p1,p2)])

        print(numpy_equivalent,test_node.value)

        assert new_node.name==test_node.name+1
        assert new_node.value==numpy_equivalent(test_node.value)
        assert new_node.back_deriv=={test_node.name: numpy_derivative(test_node.value)}
        assert new_node.for_deriv==numpy_derivative(test_node.value)*test_node.for_deriv
        assert new_node.parents==[test_node]
        assert test_node.child==[new_node]

        assert numpy_equivalent(test_number)==fun(test_number)

        with pytest.raises(TypeError):
            non_cooperating_object=[]
            fun(non_cooperating_object)

    def test_sin(self):
        self.helper_test(sin,np.sin,np.cos)
    def test_cos(self):
        self.helper_test(cos,np.cos,lambda x: -np.sin(x))
    # UPDATE DERIVATIVE DEFINIZION
    def test_tan(self):
        self.helper_test(tan,np.tan,lambda x: np.sin(x)/(np.cos(x)))



    def sinh(self):
        self.helper_test(sinh,np.sinh,np.cosh)
    def cosh(self):
        self.helper_test(cosh,np.cosh,np.sinh)
        assert 2==1
    def tanh(self):
        self.helper_test(tanh,np.tanh,lambda x: 1/(np.cosh(x)**2))
    
    def test_sqrt(self):
        self.helper_test(sqrt,np.sqrt,lambda x: 0.5*(x**(-0.5)),input_range=[0,100])
    
    def test_exp(self):
        def wrapper(base):
            def fun(var):
                return exp(var,base)
            return fun
        base=np.random.randint(1,100)   
        fun=wrapper(base)
        self.helper_test(fun,lambda x: base**x,lambda x: (base**x) * np.log(base))
    
    def test_log(self):
        def wrapper(base):
            def fun(var):
                return log(var,base)
            return fun
        base=np.random.randint(1,100)   
        fun=wrapper(base)
        self.helper_test(fun,lambda x: np.log(x)/np.log(base),lambda x: 1/(x*np.log(base)),[0.0001,100])
    
    def test_sigmoid(self):
        self.helper_test(sigmoid,lambda x: 1/(1+np.exp(-x)),lambda x:(1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x))) )

    def test_arcsin(self):
        self.helper_test(arcsin,np.arcsin,lambda x: 1/(np.sqrt(1-x**2)),input_range=[-1,1])
    def test_arccos(self):
        self.helper_test(arcos,np.arccos,lambda x: -1/(np.sqrt(1-x**2)),input_range=[-1,1])
    def test_arctan(self):
        self.helper_test(artan,np.arctan,lambda x: 1/(1+x**2))


    
    

