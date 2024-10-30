import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def sigmoid_derivative(x):
    return x*(1-x)

class layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)
        self.output = None
        self.input = None
        self.delta = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, input):
        self.input = input
        self.output = sigmoid(np.dot(input, self.weights) + self.bias)
        return self.output

    def backward(self, delta):
        self.delta = delta
        self.delta_weights = np.dot(self.input.T, delta)
        self.delta_bias = np.sum(delta, axis=0, keepdims=True)
        return np.dot(delta, self.weights.T)

    def update(self, learning_rate):
        self.weights += learning_rate * self.delta_weights
        self.bias += learning_rate * self.delta_bias

inputlayer = layer(8, 8)
hiddenlayer = layer(8, 3)
outputlayer = layer(3, 8)

input = np.array([[0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [1,0,0,0,0,0,0,0]])

output = np.array([[0,0,0,0,0,0,0,1],
                     [0,0,0,0,0,0,1,0],
                     [0,0,0,0,0,1,0,0],
                     [0,0,0,0,1,0,0,0],
                     [0,0,0,1,0,0,0,0],
                     [0,0,1,0,0,0,0,0],
                     [0,1,0,0,0,0,0,0],
                     [1,0,0,0,0,0,0,0]])

learning_rate = 0.1

# for i in range(10000):
#     for j in range(8):
#         inputlayer.forward(input[j:j+1])
#         hiddenlayer.forward(inputlayer.output)
#         outputlayer.forward(hiddenlayer.output)

#         outputlayer.backward(sigmoid_derivative(outputlayer.output) * (output[j:j+1] - outputlayer.output))
#         hiddenlayer.backward(outputlayer.delta)
#         inputlayer.backward(hiddenlayer.delta)

#         outputlayer.update(learning_rate)
#         hiddenlayer.update(learning_rate)
#         inputlayer.update(learning_rate)

#     if i % 1000 == 0:
#         print("Error: ", np.mean(np.square(output - outputlayer.output)))

print((input[1:3]))

    