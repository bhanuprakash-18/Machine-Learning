import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


class KNearestNeighbors:
    def __init__(self, k=3, p=2):
        """
        Initialize k-NN classifier.
        
        Parameters:
        - k: number of nearest neighbors to consider
        - p: parameter for Minkowski distance (1=Manhattan, 2=Euclidean, etc.)
        """
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None
        
    def minkowski_distance(self, x1, x2):
        """
        Compute Minkowski distance between two points.
        
        Special cases:
        - p=1: Manhattan distance
        - p=2: Euclidean distance
        - p=∞: Chebyshev distance (implemented with p=np.inf)
        """
        return np.sum(np.abs(x1 - x2)**self.p)**(1/self.p)
    
    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """Predict labels for test data."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        """Helper function to predict label for a single sample."""
        # Compute distances between x and all examples in the training set
        distances = [self.minkowski_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label (handle ties by reducing k)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage demonstrating the curse of dimensionality
def demonstrate_curse_of_dimensionality():
    # Create low and high dimensional datasets
    X_low, y_low = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                                      n_redundant=0, random_state=42)
    X_high, y_high = make_classification(n_samples=1000, n_features=100, n_informative=10, 
                                        n_redundant=90, random_state=42)
    
    # Split into train and test
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, random_state=42)
    X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size=0.2, random_state=42)
    
    # Train and evaluate on low dimensional data
    knn_low = KNearestNeighbors(k=5)
    knn_low.fit(X_train_low, y_train_low)
    pred_low = knn_low.predict(X_test_low)
    acc_low = accuracy_score(y_test_low, pred_low)
    
    # Train and evaluate on high dimensional data
    knn_high = KNearestNeighbors(k=5)
    knn_high.fit(X_train_high, y_train_high)
    pred_high = knn_high.predict(X_test_high)
    acc_high = accuracy_score(y_test_high, pred_high)
    
    print(f"Accuracy on low-dimensional data (d=2): {acc_low:.2f}")
    print(f"Accuracy on high-dimensional data (d=100): {acc_high:.2f}")
    print("\nNotice how performance degrades with higher dimensions due to the curse of dimensionality!")

if __name__ == "__main__":
    # Simple example
    iris = load_iris()
    iris.head()
    iris.info()
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2f}")
    
    # Demonstrate curse of dimensionality
    print("\nDemonstrating the curse of dimensionality:")
    demonstrate_curse_of_dimensionality()