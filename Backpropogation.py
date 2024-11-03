import numpy as np

class neural_network:
    def __init__(self,input,output,learning_rate):
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.inputlayer = layer(8, 3)
        self.hiddenlayer = layer(3, 8)

    def normalize(self, x):
        normalized = np.zeros_like(x)
        max_indices = np.argmax(x, axis=1)
        for i, idx in enumerate(max_indices):
            normalized[i, idx] = 1
        return normalized
    
    # Normalize the output and calculate the error, this error is not for training but for checking convergence
    def normalizederror(self,x):
        sum = 0
        for i in range(8):
            for j in range(8):
                sum += abs(x[i][j] - self.output[i][j])
        return sum

    def train(self):
        for i in range(10000):
            inputlayer_output = self.inputlayer.forward(self.input)
            final_output = self.hiddenlayer.forward(inputlayer_output)
            
            # Calculate error
            error = self.output - final_output
            
            # Backpropagation
            self.hiddenlayer.backward(error)
            self.inputlayer.backward(self.hiddenlayer.delta)
            
            # Update weights
            self.hiddenlayer.update(self.learning_rate)
            self.inputlayer.update(self.learning_rate)

            # Normalize the output and calculate the error, this error is not for training but for checking convergence
            normalized_output = self.normalize(final_output)
            normalized_error = self.normalizederror(normalized_output)
        
            if normalized_error == 0:
                return i
        return i
    

class layer:
    def sigmoid(self,x):
        return 1/(1+np.exp(-x)) 

    def sigmoid_derivative(self,x):
        return x*(1-x)
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size + 1, output_size)  # +1 for bias
        self.output = None
        self.input = None
        self.delta = None
        self.delta_weights = None

    def forward(self, input):
        # Add bias term to the input
        self.input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
        self.output = self.sigmoid(np.dot(self.input, self.weights))
        return self.output

    def backward(self, delta):
        # Calculate delta weights including bias
        self.delta_weights = np.dot(self.input.T, delta * self.sigmoid_derivative(self.output))
        self.delta = np.dot(delta, self.weights.T)[:, :-1]  # Exclude bias from delta

    def update(self, learning_rate):
        self.weights += learning_rate * self.delta_weights

        

    
    



    