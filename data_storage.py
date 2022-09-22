"""
Jimson Huang
CS 422 Project 1
"""

import numpy as np


# Takes a 2D array of string values and returns a 2D array of features and 1D array of labels
def build_nparray(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = data[1:, :-1].astype(float)
    labels = data[1:, -1].astype(int)
    return features, labels


# Take 2D arrray of string values and return a 2D list of features and 1D list of labels
def build_list(data: np.ndarray) -> tuple[list, list]:
    return tuple(arr.tolist() for arr in build_nparray(data))


# Take 2D array of string values and return a list of feature dictionary and label dictionary
def build_dict(data: np.ndarray) -> tuple[list, dict]:
    attributes = data[0, :-1]
    features, labels = build_nparray(data)

    features = [{attributes[j]: features[i][j] for j in range(len(features[i]))} for i in range(len(features))]
    labels = {i: labels[i] for i in range(len(labels))}
    return features, labels