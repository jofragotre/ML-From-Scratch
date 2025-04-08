import numpy as np

class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy Loss function.
    This class computes the categorical cross-entropy loss between true labels and predicted probabilities.
    """
    def __init__(self, epsilon=1e-15):
        """
        Initializes the CategoricalCrossEntropy with a small epsilon to avoid log(0).
        """
        self.epsilon = epsilon

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy loss.

        Parameters:
        - y_true: True labels (one-hot encoded). Shape should be (batch_size, num_classes).
        - y_pred: Predicted probabilities. Shape should be (batch_size, num_classes).

        Returns:
        - loss: Computed categorical cross-entropy loss.
        """
        
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute the categorical cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred), axis=1).mean()
        
        return loss