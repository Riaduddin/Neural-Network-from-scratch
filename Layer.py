import numpy as np

class Dense:
    def __init__(self,n_inputs,n_neurons):
        #initialized the weights & biases value
        self.weights=np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
    
    #forward pass
    def forward(self,inputs):
        #calculating the z=wx+b
        self.output=np.dot(inputs,self.weights)+self.biases
        #remember input variables
        self.inputs=inputs
    
    #backward pass
    def backward(self,dvalues):
        #Gradient on parameters
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbiases=np.sum(dvalues,axis=0,keepdims=True)

        #Gradient on inputs
        self.dinputs=np.dot(dvalues,self.weights.T)