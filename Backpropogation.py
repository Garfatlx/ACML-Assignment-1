import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def sigmoid_derivative(x):
    return x*(1-x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) 

def normalize(x):
    normalized = np.zeros_like(x)
    max_indices = np.argmax(x, axis=1)
    for i, idx in enumerate(max_indices):
        normalized[i, idx] = 1
    
    return normalized

def normalizederror(x):
    sum = 0
    for i in range(8):
        for j in range(8):
            sum += abs(x[i][j] - output[i][j])
    return sum

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

inputlayer = layer(8, 3)
hiddenlayer = layer(3, 8)

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
    inputlayer_output = inputlayer.forward(input)
    final_output = hiddenlayer.forward(inputlayer_output)
    
    # Calculate error
    error = output - final_output
    
    # Backpropagation
    hiddenlayer.backward(error)
    inputlayer.backward(hiddenlayer.delta)
    
    # Update weights
    hiddenlayer.update(learning_rate)
    inputlayer.update(learning_rate)

    # Normalize the output and calculate the error, this error is not for training but for checking convergence
    normalized_output = normalize(final_output)
    normalized_error = normalizederror(normalized_output)

   

    if normalized_error == 0:
        print("Converged at Epoch: ", i)
        break
    



    