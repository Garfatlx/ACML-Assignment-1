import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def sigmoid_derivative(x):
    return x*(1-x)

class layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size + 1, output_size)  # +1 for bias
        self.output = None
        self.input = None
        self.delta = None
        self.delta_weights = None

    def forward(self, input):
        # Add bias term to the input
        self.input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
        self.output = sigmoid(np.dot(self.input, self.weights))
        return self.output

    def backward(self, delta):
        # Calculate delta weights including bias
        self.delta_weights = np.dot(self.input.T, delta * sigmoid_derivative(self.output))
        self.delta = np.dot(delta, self.weights.T)[:, :-1]  # Exclude bias from delta

    def update(self, learning_rate):
        self.weights += learning_rate * self.delta_weights

hiddenlayer = layer(8, 3)
outputlayer = layer(3, 8)

input = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0]])

output = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0]])

learning_rate = 0.1

# Example training loop
for i in range(10000):
    hidden_output = hiddenlayer.forward(input)
    final_output = outputlayer.forward(hidden_output)
    
    # Calculate error
    error = output - final_output
    
    # Backpropagation
    outputlayer.backward(error)
    hiddenlayer.backward(outputlayer.delta)
    
    # Update weights
    outputlayer.update(learning_rate)
    hiddenlayer.update(learning_rate)

    if i % 1000 == 0:
        print("Epoch: ", i)
        print(outputlayer.output)
        print("Error: ", np.mean(np.square(output - outputlayer.output)))



    