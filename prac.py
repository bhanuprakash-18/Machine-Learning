import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd


def minkowski_distance (x1,x2,p):
    distance = np.sum(np.abs(x1-x1)**p)**(1/p)
    return distance

def predict(x_train,y_train,x_test,k,p):

    distances = [minkowski_distance(x_test,x_train.iloc[i],p) for i in range(len(x_train))]
    k_indices = np.argsort(distances)[:k]
    predictions = y_train.iloc[k_indices]
    return predictions

if __name__ == "__main__":
    iris_data = load_iris()
    iris_features = iris_data.features
    

