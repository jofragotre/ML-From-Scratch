import numpy as np

class Module:
    """
    Base class for all layers in the neural network.
    This class can be extended to create custom layers.
    """
    def __init__(self, eval=False):
        self.eval = eval
        self.inputs = None

    def forward(self, inputs: np.ndarray):
        """
        Forward pass through the layer. To be implemented by subclasses.
        """
        if not self.eval:
            self.inputs = inputs
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, d_output: np.ndarray):
        """
        Backward pass through the layer. To be implemented by subclasses.
        """
        raise NotImplementedError("Backward method not implemented.")

class Neuron(Module):
    """
    A simple Neuron class that represents a single neuron in a neural network.
    It initializes weights and a bias for the neuron.
    """
    def __init__(self, num_inputs: int, initialization='kaim', **kwargs):
        """
        Initializes the Neuron with a specified number of inputs and weight initialization method.
        """

        super().__init__(**kwargs)

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
        
        # Initialize gradients for weights and bias
        self.d_weights = None
        self.d_bias = None
    
    def forward(self, inputs: np.ndarray):
        """
        Computes the output of the neuron given the inputs.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)  # Reshape to 2D if it's a single sample (batch dim of 1)
        if inputs.shape[1] != self.num_inputs:
            raise ValueError(f"Expected inputs with {self.num_inputs} features, but got {inputs.shape[1]}.")

        # Save inputs for backward pass if not eval
        if not self.eval:
            self.inputs = inputs
        
        # Compute the output of the neuron
        # weights shape: (num_inputs,)
        # inputs shape: (batch_size, num_inputs)
        # bias shape: (1,)
        # output shape: (batch_size,)

        result = inputs @ self.weights.T + self.bias

        return result

    def backward(self, d_output: np.ndarray):
        """
        Compute gradients for weights and bias given the gradient of the output
        Args:
            d_output: shape (batch_size,)
            inputs: shape (batch_size, num_inputs)
        Returns:
            gradient with respect to inputs
        """
        inputs = self.inputs  # Retrieve saved inputs
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            d_output = d_output.reshape(1)

        # Compute per-sample weight gradients
        grad_weights_per_sample = inputs * d_output[:, np.newaxis]  # shape: (batch_size, num_inputs)

        # Sum over batch to get final weight and bias gradients
        self.d_weights = np.sum(grad_weights_per_sample, axis=0)  # shape: (num_inputs,)
        self.d_bias = np.sum(d_output)  # scalar

        # Compute gradient w.r.t. inputs
        d_inputs = d_output[:, np.newaxis] * self.weights  # shape: (batch_size, num_inputs)

        return d_inputs


class LinearLayer(Module):
    """
    A simple LinearLayer class that represents a LinearLayer in a neural network.
    It contains multiple neurons and computes the output for the entire LinearLayer.
    """
    def __init__(self, num_neurons, num_inputs, initialization='kaim', **kwargs):
        """
        Initializes the LinearLayer with a specified number of neurons and weight initialization method.
        """
        super().__init__(**kwargs)

        self.neurons = [Neuron(num_inputs, initialization) for _ in range(num_neurons)]

        # Store weights and biases for all neurons in the LinearLayer. This will allow for efficient matrix multiplication.

        # Shape of W: (num_neurons, num_inputs)
        self.weights = np.array([neuron.weights for neuron in self.neurons]) 

        # Shape of b: (num_neurons,)
        self.biases = np.array([neuron.bias for neuron in self.neurons]).squeeze() # Squeeze to remove extra dimension

        # Initialize gradients for weights and biases
        self.d_weights = None
        self.d_biases = None
    
    def forward(self, inputs: np.ndarray):
        """
        Computes the output of the LinearLayer given the inputs.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.shape[1] != self.weights.shape[1]:
            raise ValueError(f"Expected inputs with {self.weights.shape[1]} features, but got {inputs.shape[1]}.")
        
        # Save inputs for backward pass if not eval
        if not self.eval:
            self.inputs = inputs
        
        # Compute the output of the LinearLayer
        # weights shape: (num_neurons, num_inputs)
        # inputs shape: (batch_size, num_inputs)
        # biases shape: (num_neurons,)
        # output shape: (batch_size, num_neurons)
        # Using matrix multiplication for efficiency

        output = inputs @ self.weights.T + self.biases

        return output

    def backward(self, d_output: np.ndarray):
        """
        Compute gradients for the linear layer
        Args:
            d_output: gradient of shape (batch_size, num_neurons)
            inputs: inputs of shape (batch_size, num_inputs)
        Returns:
            gradient with respect to inputs
        """

        inputs = self.inputs  # Retrieve saved inputs

        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            d_output = d_output.reshape(1, -1)

        # Compute gradients for weights and biases
        # d_output shape: (batch_size, num_neurons)
        # d_inputs shape: (batch_size, num_inputs)
        
        # d_L/d_weights = d_output.T @ inputs
        # d_weights shape: (num_neurons, num_inputs)
        self.d_weights = d_output.T @ inputs

        # d_biases shape: (num_neurons,)
        self.d_biases = np.sum(d_output, axis=0)
        
        # Compute gradient with respect to inputs
        # Shape: (batch_size, num_inputs)
        d_inputs = d_output @ self.weights
        
        return d_inputs


class ReLu(Module):
    """
    A simple ReLU activation function class.
    It applies the ReLU activation function to the input.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: np.ndarray):
        """
        Applies the ReLU activation function. Element-wise.
        """
        if not self.eval:
            self.inputs = x

        # ReLU activation: max(0, x)
        return np.maximum(0, x)

    def backward(self, d_output: np.ndarray):
        """
        Compute gradient for ReLU activation
        Args:
            d_output: gradient from next layer
            inputs: original inputs to ReLU
        Returns:
            gradient with respect to inputs
        """
        inputs = self.inputs  # Retrieve saved inputs

        # ReLU gradient: 1 if input > 0, 0 otherwise
        d_inputs = d_output * (inputs > 0)
        return d_inputs


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

    ### Backward pass is easier when combined with cross entropy loss.    

if __name__ == "__main__":

    # Example neuron usage
    neuron = Neuron(num_inputs=4, initialization='kaim')
    inputs = np.random.rand(10,4)  # 10 samples, 4 features
    # Forward pass through the neuron
    output = neuron.forward(inputs)
    print(f"Neuron output: {output.shape}")

    # Example LinearLayer usage
    linearlayer = LinearLayer(num_neurons=3, num_inputs=4, initialization='kaim')
    linearlayer_output = linearlayer.forward(inputs)
    print(f"LinearLayer output: {linearlayer_output.shape}")

    # Example ReLU usage
    relu = ReLu()
    relu_output = relu.forward(linearlayer_output)
    print(f"ReLU output: {relu_output.shape}")

    # Example Softmax usage
    softmax = Softmax()
    softmax_output = softmax.forward(relu_output)
    print(f"Softmax output: {softmax_output.shape}")
    
    # Test backward pass
    print("\nTesting backward pass:")
    # Assume some upstream gradient
    upstream_gradient = np.random.randn(10, 3)
    
    # Test relu backward
    d_relu = relu.backward(upstream_gradient, linearlayer_output)
    print(f"ReLU gradient shape: {d_relu.shape}")
    
    # Test linear layer backward
    d_linear = linearlayer.backward(d_relu, inputs)
    print(f"Linear layer gradient shapes:")
    print(f"- d_inputs: {d_linear.shape}")
    print(f"- d_weights: {linearlayer.d_weights.shape}")
    print(f"- d_biases: {linearlayer.d_biases.shape}")