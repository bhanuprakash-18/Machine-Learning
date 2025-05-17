from sklearn.datasets import load_iris
import pandas as pd

def preprocess_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

