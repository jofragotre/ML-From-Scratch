import numpy as np

class Neuron:
    """
    A simple Neuron class that represents a single neuron in a neural network.
    It initializes weights and a bias for the neuron.
    """
    def __init__(self, num_inputs: int, initialization='kaim'):
        """
        Initializes the Neuron with a specified number of inputs and weight initialization method.
        """
        if initialization not in ['random', 'zeros', 'kaim']:
            raise ValueError("Invalid initialization method. Choose 'random', 'zeros', or 'kaim'.")
        
        self.initialization = initialization
        self.num_inputs = num_inputs

        if self.initialization == 'zeros':
            # Zeros initialization
            # This is not a good practice for weights, but it's included for completeness
            self.weights = np.zeros(num_inputs)
            self.bias = np.zeros(1)
        elif self.initialization == 'kaim':
            # Kaiming initialization
            self.weights = np.random.randn(num_inputs) * np.sqrt(2. / num_inputs)
            self.bias = np.zeros(1)
        else:
            # Random initialization of weights
            np.random.seed(42)
            self.weights = np.random.rand(num_inputs)
            self.bias = np.random.rand(1)
    
    def forward(self, inputs: np.ndarray):
        """
        Computes the output of the neuron given the inputs.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)  # Reshape to 2D if it's a single sample (batch dim of 1)
        if inputs.shape[1] != self.num_inputs:
            raise ValueError(f"Expected inputs with {self.num_inputs} features, but got {inputs.shape[1]}.")
        
        # Compute the output of the neuron
        # weights shape: (num_inputs,)
        # inputs shape: (batch_size, num_inputs)
        # bias shape: (1,)
        # output shape: (batch_size,)

        result = inputs @ self.weights.T + self.bias

        return result
    

class LinearLayer:
    """
    A simple LinearLayer class that represents a LinearLayer in a neural network.
    It contains multiple neurons and computes the output for the entire LinearLayer.
    """
    def __init__(self, num_neurons, num_inputs, initialization='kaim'):
        """
        Initializes the LinearLayer with a specified number of neurons and weight initialization method.
        """
        self.neurons = [Neuron(num_inputs, initialization) for _ in range(num_neurons)]

        # Store weights and biases for all neurons in the LinearLayer. This will allow for efficient matrix multiplication.

        # Shape of W: (num_neurons, num_inputs)
        self.weights = np.array([neuron.weights for neuron in self.neurons]) 

        # Shape of b: (num_neurons,)
        self.biases = np.array([neuron.bias for neuron in self.neurons]).squeeze() # Squeeze to remove extra dimension
    
    def forward(self, inputs: np.ndarray):
        """
        Computes the output of the LinearLayer given the inputs.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.shape[1] != self.weights.shape[1]:
            raise ValueError(f"Expected inputs with {self.weights.shape[1]} features, but got {inputs.shape[1]}.")
        
        # Compute the output of the LinearLayer
        # weights shape: (num_neurons, num_inputs)
        # inputs shape: (batch_size, num_inputs)
        # biases shape: (num_neurons,)
        # output shape: (batch_size, num_neurons)
        # Using matrix multiplication for efficiency

        output = inputs @ self.weights.T + self.biases

        return output


class ReLu:
    """
    A simple ReLU activation function class.
    It applies the ReLU activation function to the input.
    """
    def __init__(self):
        pass
    def forward(self, x: np.ndarray):
        """
        Applies the ReLU activation function. Element-wise.
        """
        return np.maximum(0, x)


class Softmax:
    """
    A simple Softmax activation function class.
    It applies the Softmax activation function to the input.
    """
    def __init__(self):
        pass
    def forward(self, inputs: np.ndarray):
        """
        Applies the Softmax activation function. Element-wise.
        """
        # Exponentiate the input. Shape: (batch_size, num_classes). Subtract max for numerical stability.
        e_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Sum across classes. Shape: (batch_size, 1)
        sum_e_x = np.sum(e_x, axis=1, keepdims=True)  
        
        # Normalize to get probabilities
        # Shape: (batch_size, num_classes) 
        # Broadcasting will handle the shape of sum_e_x
        result = e_x / sum_e_x  
        return result
    

if __name__ == "__main__":

    # Example neuron usage
    neuron = Neuron(num_inputs=4, initialization='kaim')
    inputs = np.random.rand(10,4)  # 10 samples, 4 features
    # Forward pass through the neuron
    output = neuron.forward(inputs)
    print(f"Neuron output: {output}")

    # Example LinearLayer usage
    linearlayer = LinearLayer(num_neurons=3, num_inputs=4, initialization='kaim')
    linearlayer_output = linearlayer.forward(inputs)
    print(f"LinearLayer output: {linearlayer_output}")

    # Example ReLU usage
    relu = ReLu()
    relu_output = relu.forward(linearlayer_output)
    print(f"ReLU output: {relu_output}")

    # Example Softmax usage
    softmax = Softmax()
    softmax_output = softmax.forward(relu_output)
    print(f"Softmax output: {softmax_output}")