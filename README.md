# Neural Network Implementation from Scratch

A Python implementation of a neural network from scratch using only NumPy. This project includes implementations of fundamental neural network components and demonstrates their use with the Iris dataset.

## Project Structure

```
ml-from-scratch/
├── layers.py        # Neural network layer implementations
├── network.py       # Neural network class
├── iris_example.py  # Example using Iris dataset
└── train_example.py # Basic training example
```

## Components

### Layers (`layers.py`)
- `Module`: Base class for all layers
- `Neuron`: Single neuron implementation
- `LinearLayer`: Fully connected layer
- `ReLU`: Rectified Linear Unit activation
- `Softmax`: Softmax activation for classification

### Network (`network.py`)
- Neural network implementation supporting multiple layers
- Forward and backward propagation
- Gradient descent optimization

## Features

- **Weight Initialization Options**:
  - Kaiming initialization
  - Zero initialization
  - Random initialization

- **Modular Design**:
  - Easy to add new layer types
  - Flexible network architecture

- **Built-in Support**:
  - Batch processing
  - Automatic gradient computation
  - Learning rate adjustment

## Usage

### Basic Example
```python
from network import Network
from loss_functions import CategoricalCrossEntropy

# Create network (4 inputs, 8 hidden neurons, 3 outputs)
network = Network([4, 8, 3])

# Training
predictions = network.forward(X_train)
loss = cross_entropy_loss.forward(y_train, predictions)
loss_gradient = cross_entropy_loss.backward(y_train, predictions)
network.backward(loss_gradient)
network.update(learning_rate=0.01)
```

### Iris Dataset Example
```python
# Run the Iris dataset example
python iris_example.py
```

## Technical Details

### Forward Propagation
- Input processing
- Layer-by-layer computation
- Activation functions application

### Backward Propagation
- Gradient computation
- Chain rule application
- Parameter updates

### Layer Operations
- Linear transformation: y = Wx + b
- ReLU activation: f(x) = max(0, x)
- Softmax: σ(x)ᵢ = exp(xᵢ) / Σⱼexp(xⱼ)

## Requirements

- NumPy
- scikit-learn (for Iris dataset example)

## TODO:
[ ] - Create Param class that stores weight and gradient data. Replace it on all other classes.

[ ] - Create optimizer class to update Params based on their gradient data.

[ ] - Support more common layer types: Conv2d, BatchNorm, Sigmoid, etc..

## License

MIT License
