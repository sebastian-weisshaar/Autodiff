import sys
import numpy as np
import pytest
sys.path.insert(1, './src/autodiff')
from functions import sin,cos,tan,arcsin,arccos,arctan,sinh,cosh,tanh,sqrt,exp,log,sigmoid
from autodiff import Node



class TestFunction:
    """ Class to test functions in functions.py
    
    Methods
    -------
    create_node
    helper_test
    test_sin
    test_cos
    test_tan
    test_sinh
    test_cosh
    test_tanh
    test_arcsin
    test_arccos
    test_arctan 
    test_sqrt
    test_log
    test_exp
    test_sigmoid
    
    """
    
    def create_node(self,value_range=[-100,100]):
        """Function to create a test node

        Parameters
        ----------
        value_range: list
            list with lower and upper value for the range of the values

        Returns
        -------
        Node
            A node initiliazed with a random value in the specified range, a random derivative
        """
        p1,p2=value_range
        value=np.random.choice([np.random.uniform(p1,p2),np.random.randint(p1,p2)])
        for_deriv=np.random.choice([np.random.normal(12,10),np.random.randint(-200,300)])
        test_node=Node(np.random.randint(-1,1000),value,[],[],for_deriv,{})
        return test_node

    def helper_test(self,fun,number_equivalent,derivative,input_range=[-100,100],derivative_range=[-100,100]):
        """ Helper function to create a test for a function

        The function takes in the function from the functions.py file and checks whether it performs correctly on nodes and values

        Parameters
        ----------

        fun: function
            The imported function from functions.py
        number equivalent: function
            A simple function that is the equivalent of the imported function on integers or floats
        derivative: function
            The derivative of the imported function if applied on integers or floats
        input_range: list
            A list containing the lower and upper end of possible values for the function, domain of function
        derivative range: list
            A list containing the lower and upper end of possible values for the derivative, domain of derivative

        Returns
        -------
        None

        """
       
       
        test_node=self.create_node(value_range=input_range)
        new_node=fun(test_node)
        p1,p2=input_range
        test_number1=np.random.uniform(p1,p2)
        test_number2=np.random.randint(p1,p2)

        assert new_node.name==test_node.name+1
        assert new_node.value==number_equivalent(test_node.value)
        deriv=np.round(derivative(test_node.value),10)
        assert np.round(new_node.back_deriv[test_node.name],10)==deriv
        assert type(new_node.back_deriv)==dict
        #assert new_node.back_deriv=={test_node.name: numpy_derivative(test_node.value)}
        assert new_node.for_deriv==derivative(test_node.value)*test_node.for_deriv
        assert new_node.parents==[test_node]
        assert test_node.child==[new_node]

        assert number_equivalent(test_number1)==fun(test_number1)
        assert number_equivalent(test_number2)==fun(test_number2)

        with pytest.raises(TypeError):
            non_cooperating_object=[]
            fun(non_cooperating_object)


    def test_sin(self):
        """Apply helper_test on the sin function"""
        self.helper_test(sin,np.sin,np.cos)
    def test_cos(self):
        """Apply helper_test on the cos function"""
        self.helper_test(cos,np.cos,lambda x: -np.sin(x))
    # UPDATE DERIVATIVE DEFINITION
    def test_tan(self):
        """Apply helper_test on the tan function"""
        self.helper_test(tan,np.tan,lambda x: np.sin(x)/(np.cos(x)))

    def test_sinh(self):
        """Apply helper_test on the sinh function"""
        self.helper_test(sinh,np.sinh,np.cosh)
    def test_cosh(self):
        """Apply helper_test on the cosh function"""
        self.helper_test(cosh,np.cosh,np.sinh)
    def test_tanh(self):
        """Apply helper_test on the tanh function"""
        self.helper_test(tanh,np.tanh,lambda x: (1/np.cosh(x))**2)
    
    def test_sqrt(self):
        """Apply helper_test on the sqrt function"""
        self.helper_test(sqrt,np.sqrt,lambda x: 0.5*(x**(-0.5)),input_range=[0,100])
    
    def test_exp(self):
        """Apply helper_test on the sqrt function"""
        def wrapper(base):
            def fun(var):
                return exp(var,base)
            return fun
        with pytest.raises(TypeError):
            wrong_fun=wrapper(-2)
            number=np.random.uniform(0.0001,100)
            wrong_fun(number)
        base=np.random.randint(1,100)   
        fun=wrapper(base)
        self.helper_test(fun,lambda x: base**x,lambda x: (base**x) * np.log(base))
    
    def test_log(self):
        """Apply helper_test on the log function"""
        def wrapper(base):
            def fun(var):
                return log(var,base)
            return fun
        
        with pytest.raises(TypeError):
            wrong_fun=wrapper(-2)
            number=np.random.uniform(0.0001,100)
            wrong_fun(number)
        base=np.random.randint(1,100)   
        fun=wrapper(base)
        self.helper_test(fun,lambda x: np.log(x)/np.log(base),lambda x: 1/(x*np.log(base)),[0.0001,100])
    
    def test_sigmoid(self):
        """Apply helper_test on the sigmoid function"""
        self.helper_test(sigmoid,lambda x: 1/(1+np.exp(-x)),lambda x: ((np.exp(-x))/((np.exp(-x)+1)**2)) )

    def test_arcsin(self):
        """Apply helper_test on the test_arcsin function"""
        self.helper_test(arcsin,np.arcsin,lambda x: 1/(np.sqrt(1-x**2)),input_range=[-1,1])
    def test_arccos(self):
        """Apply helper_test on the arccos function"""
        self.helper_test(arccos,np.arccos,lambda x: -1/(np.sqrt(1-x**2)),input_range=[-1,1])
    def test_arctan(self):
        """Apply helper_test on the arctan function"""
        self.helper_test(arctan,np.arctan,lambda x: 1/(1+x**2))


     
    

