import functools
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    class_column = 'Survived'
    raw_data = pd.read_csv(r'./data/train.csv')
    processed_data = raw_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    processed_data['Age'] = processed_data['Age'].apply(distribute, mean=processed_data['Age'].mean(), std=processed_data['Age'].std())
    processed_data['Fare'] = processed_data['Fare'].apply(distribute, mean=processed_data['Fare'].mean(), std=processed_data['Fare'].std())
    processed_data = processed_data.rename(index=str, columns={'Age':'AgeClass', 'Fare':'FareClass'})
    processed_data = pd.get_dummies(processed_data, prefix=['Sex', 'Embarked'], drop_first=False)
    processed_data = processed_data.fillna(processed_data.mean())

    probabilities_died = calculate_probabilities(processed_data, class_column, 0)
    probabilities_survived = calculate_probabilities(processed_data, class_column, 1)

    ps, fs = len(probabilities_survived) / len(processed_data), len(probabilities_died) / len(processed_data)

    processed_data['Prediction'] = processed_data.drop('Survived', axis=1).apply(lambda fs: predict(fs, ps, fs, probabilities_survived, probabilities_died), axis=1)
    processed_data['Correct'] = np.where(processed_data['Survived'] == processed_data['Prediction'], 1, 0)
    
    accuracy = processed_data['Correct'].sum() / len(processed_data)
    print(f'Accuracy: {accuracy}')
    plt.show()


def predict(features, ps, pf, fpgs, fpgf):
    # Probability for features given success: P(f0...fn | class = 1)
    pfgs = functools.reduce(operator.mul, [fpgs[i][features[i]] if features[i] in fpgs[i] else (1 / len(fpgs[i])) for i in range(len(features))])
    # Probability for features given failure: P(f0...fn | class = 0)
    pfgf = functools.reduce(operator.mul, [fpgf[i][features[i]] if features[i] in fpgf[i] else (1 / len(fpgf[i])) for i in range(len(features))])

    probability_success = pfgs * ps / ((pfgs * ps) + (pfgf * pf))
    probability_failure = pfgf * pf / ((pfgf * pf) + (pfgs * ps))

    return 1 if probability_success[0] > probability_failure[0] else 0

def calculate_probabilities(data, class_column, classification=1):
    subset = data[data[class_column] == classification].drop([class_column], axis=1)
    counts = [subset[column].value_counts().to_dict() for column in subset]

    return [{k: (v + 1) / (sum(d.values()) + len(d)) for k, v in d.items()} for d in counts]

def distribute(v, mean, std):
    if v < mean - std:
        return -1
    if mean - std <= v <= mean + std:
        return 0
    return 1

if __name__ == "__main__":
    main()
