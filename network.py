import numpy as np
from layers import LinearLayer, ReLu, Softmax

class Network:
    def __init__(self, layer_sizes, eval=False):
        """
        Initialize a neural network with specified layer sizes.
        layer_sizes: list of integers representing the size of each layer
        """
        self.layers = []
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LinearLayer(layer_sizes[i+1], layer_sizes[i], eval=eval))
            if i < len(layer_sizes) - 2:  # No ReLU after last layer
                self.layers.append(ReLu(eval=eval))
        
    def forward(self, X):
        """
        Forward pass through the network.
        Stores intermediate activations for backpropagation.
        """
        current_input = X
        
        for layer in self.layers:
            current_output = layer.forward(current_input)
            current_input = current_output
            
        return current_input
    
    def backward(self, loss_gradient):
        """
        Backward pass through the network.
        Computes gradients for all layers.
        """
        current_gradient = loss_gradient
        
        # Iterate through layers in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            # Input to current layer is the activation from previous layer
            current_gradient = layer.backward(current_gradient)
    
    def update(self, learning_rate):
        """
        Update network parameters using computed gradients.
        """
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                # Update weights and biases
                layer.weights -= learning_rate * layer.d_weights
                layer.biases -= learning_rate * layer.d_biases