import numpy as np

class ReLU:
    def forward(self,inputs):
        #remember input variables
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
    
    def backward(self,dvalues):

        self.dinputs=dvalues.copy()
        #zero gradient on negative inputs
        self.dinputs[self.inputs<=0]=0