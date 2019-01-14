import numpy as np
import pandas as pd

from sklearn import svm, model_selection

def main():
    label_column = 'Survived'
    raw_data = pd.read_csv(r'./data/train.csv')
    processed_data = raw_data.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)
    processed_data.dropna(inplace=True)
    # Check against dropfirst = False
    processed_data = pd.get_dummies(processed_data, ['Sex', 'Embarked'], drop_first=True)

    train_points, test_points, train_labels, test_labels = model_selection.train_test_split(processed_data.drop([label_column], axis=1), processed_data[label_column], test_size=0.15)

    model = svm.LinearSVC(max_iter=30000)
    model.fit(train_points, train_labels)
    print(model.score(test_points, test_labels))

if __name__ == "__main__":
    main()