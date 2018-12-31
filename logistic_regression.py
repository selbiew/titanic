import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    target_column = 'Survived'
    raw_data = pd.read_csv(r'./data/train.csv')
    processed_data = raw_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    processed_data = pd.get_dummies(processed_data, prefix=['Sex', 'Embarked'], drop_first=False)
    processed_data = processed_data.fillna(processed_data.mean())
    processed_data['bias'] = 1.0

    train_data = processed_data.iloc[:800, :]

    test_data = processed_data.iloc[800:, :]
    test_features = test_data.drop([target_column], axis=1)
    test_targets = test_data[[target_column]]

    logistic_function = sigmoid
    weights = gradient_ascent(train_data, target_column, logistic_function, batch_size=len(train_data), epochs=10000)
    hypotheses = np.rint(logistic_function(np.matmul(test_features, weights)))
    results = pd.DataFrame(np.hstack((hypotheses, test_targets)))
    accuracy = calculate_accuracy(results)

    print(accuracy)

def gradient_ascent(data, target_column, logistic_function, batch_size=1, learning_rate=0.0000001, epochs=10000):
    weights = np.zeros((data.shape[1] - 1, 1))

    for _ in range(epochs):
        sample = data.sample(n=batch_size)
        features = sample.drop(target_column, 1)
        targets = sample[[target_column]]
        hypotheses = logistic_function(np.matmul(features, weights))
        gradient = np.matmul(features.T, targets - hypotheses)
        weights += learning_rate * gradient

    return weights

def calculate_log_likelihood(scores, targets):
    return np.sum(targets * scores - np.log(1 + np.exp(scores)))

# Logistic Regression
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

# Perceptron
def binary_step(scores, h0=0):
    return np.heaviside(scores, h0)

def calculate_accuracy(results):
    return 1.0 - sum(abs(results[0] - results[1])) / len(results)

if __name__ == "__main__":
    main()