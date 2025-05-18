import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


class KNearestNeighbors:
    def __init__(self, k, p=2):
        """
        Initialize k-NN classifier.
        
        Parameters:
        - k: number of nearest neighbors to consider
        - p: parameter for Minkowski distance (1=Manhattan, 2=Euclidean, etc.). Here we use default euclidean distance p=2
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
        
    def predict(self, X_test):
        """Predict labels for test data."""
        predictions = [self._predict(X_test.iloc[i]) for i in range(len(X_test))]
        return np.array(predictions)
    
    def _predict(self, x):
        """Helper function to predict label for a single sample."""
        # Compute distances between x and all examples in the training set
        distances = [self.minkowski_distance(x, self.X_train.iloc[i]) for i in range(len(self.X_train))]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = y_train.iloc[k_indices]
        
        # Return the most common class label (handle ties by reducing k)
        most_common = k_nearest_labels.mode()
        return most_common
def error_rate(y_true, y_pred):
    """Calculate the error rate."""
    return np.mean(y_true != y_pred)

if __name__ == "__main__":
    # Load the iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    X, y = iris_df.iloc[:, :-1], iris_df['target']
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNearestNeighbors(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test).reshape(-1)

    print(np.array(y_test).reshape(-1),"\n", predictions.reshape(-1))
    error_rate = error_rate(y_test, predictions)
    print(f"Error rate: {error_rate:.2f}")
    
    # Write results to results.txt
    with open("KNN/results/results.txt", "w") as f:
        f.write("True labels:\n")
        f.write(np.array2string(np.array(y_test).reshape(-1)))
        f.write("\nPredictions:\n")
        f.write(np.array2string(predictions.reshape(-1)))
        f.write(f"\nError rate: {error_rate:.2f}\n")
    