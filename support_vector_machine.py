import numpy as np
import pandas as pd

class SupportVectorMachine:
    def __init__(self, max_iterations=5000, C=0.1):
        self.alphas = None
        self.weights = None
        self.intercept = None
        self.C = C
        self.max_iterations = max_iterations

    def fit(self, points, labels, learning_rate=0.01):
        data = np.hstack((points, labels))
        weights = np.random.rand(1, len(points[0]))
        for t in range(self.max_iterations):
            np.random.shuffle(data)
            for i in range(len(data)):
                point, label = np.atleast_2d(data[i, :len(data[i]) - 1]), data[i, -1]
                functional_margin = np.asscalar(label * weights @ point.T)
                if functional_margin <= 1:
                    weights = (1 - (learning_rate / (1 + t))) * weights + learning_rate * self.C * label * point
                else:
                    weights = (1 - (learning_rate / (1 + t))) * weights

        self.weights = weights

    def predict(self, point):
        return 1 if np.asscalar(self.weights @ point.T) >=0 else -1

def main():
    label_column = 'Survived'
    raw_data = pd.read_csv(r'./data/train.csv')
    processed_data = raw_data.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)
    processed_data.dropna(inplace=True)
    processed_data = pd.get_dummies(processed_data, ['Sex', 'Embarked'], drop_first=True)
    processed_data['bias'] = 1.0
    processed_data[label_column] = processed_data[label_column].apply(lambda n: -1 if n == 0 else 1)

    points, labels = processed_data.drop([label_column], axis=1).values, processed_data[label_column].values
    model = SupportVectorMachine()
    model.fit(points, np.atleast_2d(labels).T)

    processed_data['prediction'] = processed_data.drop([label_column], axis=1).apply(lambda point: model.predict(np.atleast_2d(point)), axis=1)
    processed_data['correct'] = np.where(processed_data[label_column] == processed_data['prediction'], 1, 0)

    accuracy = processed_data['correct'].mean()
    print(f'Accuracy: {accuracy}')
    print(processed_data.head())


if __name__ == "__main__":
    main()