import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Perceptron():
    def __init__(self):
        self.weights = None
        self.correct = 0
        self.incorrect = 0

    def feed(self, point, label):
        if self.weights is None:
            self.weights = np.zeros(point.shape)
        hypothesis = self.g(self.weights.T @ point)

        if hypothesis != label:
            self.weights += np.asscalar(label) * point
            self.incorrect += 1
        else:
            self.correct += 1
        
        return hypothesis

    def g(self, z):
        return 1 if z <= 0 else -1
    
    @property
    def accuracy(self):
        return self.correct / self.incorrect

def main():
    target_column = 'Survived'
    raw_data = pd.read_csv(r'./data/train.csv')
    processed_data = raw_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    processed_data = pd.get_dummies(processed_data, prefix=['Sex', 'Embarked'], drop_first=False)
    processed_data = processed_data.fillna(processed_data.mean())
    processed_data[target_column].map({0: -1, 1: 1})
    processed_data['bias'] = 1.0

    points, labels = processed_data.drop([target_column], axis=1).values, np.atleast_2d(processed_data[target_column].values).T

    model = Perceptron()
    for _ in range(1):            
        for i in range(len(points)):
            model.feed(points[i], labels[i])

    print(model.accuracy)

if __name__ == "__main__":
    main()