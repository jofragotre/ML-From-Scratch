import numpy as np
from network import Network
from loss_functions import CategoricalCrossEntropy

# Create sample data
np.random.seed(42)
X = np.random.randn(100, 4)  # 100 samples, 4 features
y = np.random.randint(0, 3, 100)  # 3 classes
y_one_hot = np.eye(3)[y]  # Convert to one-hot encoding

# Create network with architecture: 4 -> 8 -> 3
network = Network([4, 8, 3])

# Training parameters
learning_rate = 0.01
epochs = 100

cross_entropy_loss = CategoricalCrossEntropy()

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = network.forward(X)
    
    # Compute loss and gradients
    loss = cross_entropy_loss.forward(y_one_hot, predictions)
    loss_gradient = cross_entropy_loss.backward(y_one_hot, predictions)
    
    # Backward pass
    network.backward(loss_gradient)
    
    # Update weights
    network.update(learning_rate)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Test prediction
test_sample = np.random.randn(1, 4)
prediction = network.forward(test_sample)
print("\nTest prediction:", prediction)
print("Predicted class:", np.argmax(prediction))
