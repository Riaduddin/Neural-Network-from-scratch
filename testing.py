from Layer import Dense
from activation import ReLU,Softmax

# x=Dense(2,3)
# print(x.weights)
# print(x.biases)

a=[1,2,-3,5,-3,45,0]
b=[[1,2,-3,5,-3,45,0]]

x=ReLU()
x.forward(a)
y=Softmax()
y.forward(b)

print(x.output)
print(y.output)