from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from network import Network
from loss_functions import CategoricalCrossEntropy

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to one-hot encoding
y_one_hot = np.eye(3)[y]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create network (4 input features -> 8 hidden -> 3 output classes)
network = Network([4, 10, 3])

# Training parameters
learning_rate = 0.001
epochs = 1000

cross_entropy_loss = CategoricalCrossEntropy()

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = network.forward(X_train)
    
    # Compute loss and gradients
    loss = cross_entropy_loss.forward(y_train, predictions)
    loss_gradient = cross_entropy_loss.backward(y_train, predictions)
    
    # Backward pass
    network.backward(loss_gradient)
    
    # Update weights
    network.update(learning_rate)
    
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        # Calculate accuracy
        train_predictions = np.argmax(predictions, axis=1)
        train_true = np.argmax(y_train, axis=1)
        accuracy = np.mean(train_predictions == train_true)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Test the model
test_predictions = network.forward(X_test)
test_pred_classes = np.argmax(test_predictions, axis=1)
test_true_classes = np.argmax(y_test, axis=1)
test_accuracy = np.mean(test_pred_classes == test_true_classes)

print("\nTest Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print some example predictions
print("\nExample predictions:")
for i in range(5):
    true_class = iris.target_names[test_true_classes[i]]
    pred_class = iris.target_names[test_pred_classes[i]]
    print(f"True: {true_class}, Predicted: {pred_class}")
