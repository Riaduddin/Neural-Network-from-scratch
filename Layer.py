import numpy as np

class Dense:
    def __init__(self,n_inputs,n_neurons):
        #initialized the weights & biases value
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        #calculating the z=wx+b
        self.output=np.dot(inputs,self.weights)+sefl.biases 