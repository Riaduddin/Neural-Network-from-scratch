from Layer import Dense
from activation import ReLU
from loss import Activation_Softmax_Loss_Categoricalcrossentropy

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# x=Dense(2,3)
# print(x.weights)
# print(x.biases)
# a=[1,2,-3,5,-3,45,0]
# b=[[1,2,-3,5,-3,45,0]]
# softmax_output=[[.2,.3,.7],[.3,.2,.5],[.2,.7,.3]]
# x=ReLU()
# x.forward(a)
# y=Softmax()
# y.forward(b)
# print(x.output)
# print(y.output)

X,y=spiral_data(samples=100,classes=3)

dense1=Dense(2,3)
activation1=ReLU()
dense2=Dense(3,4)
activation2=ReLU()
dense3=Dense(4,3)

loss_activation=Activation_Softmax_Loss_Categoricalcrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
print(dense1.weights)


loss=loss_activation.forward(dense3.output,y)
#print(loss_activation.output[:5])
print('loss:',loss)

predictions=np.argmax(loss_activation.output,axis=1)
if len(y.shape)==2:
    y=np.argmax(y,axis=1)
accuracy=np.mean(predictions==y)

print('acc:',accuracy)

loss_activation.backward(loss_activation.output,y)
dense3.backward(loss_activation.dinputs)
activation2.backward(dense3.dinputs)
dense2.backward(activation2.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
print(dense3.dweights)
print(dense3.dbiases)