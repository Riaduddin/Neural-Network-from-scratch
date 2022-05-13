import numpy as np

class ReLU:
    def forward(self,inputs):
        self.output=np.maximum(0,inputs)

class Softmax:
    def forward(self,inputs):
        #substracting from the largest value to prevent dead neurons & exploding values
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))

        probabilities=exp_values-np.sum(exp_values,axis=1,keepdims=True)

        self.output=probabilities