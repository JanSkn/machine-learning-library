# Example for classification

# allows import from different folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import numpy as np
from pylearn import GaussianNaiveBayes, accuracy, precision, recall, f1_score

# Due to the short training duration, this example doesn't store and load the trained model and does training again every execution

data = pd.read_csv("examples/data/breast_cancer_data.csv")

x_train, x_test = data.iloc[:, :5], data.iloc[:25, :5]
y_train, y_test = pd.DataFrame(data.iloc[:, -1]), pd.DataFrame(data.iloc[:25, -1])

nb = GaussianNaiveBayes()
nb.fit(x_train, y_train)  
prediction = nb.predict(x_test)
prediction = pd.DataFrame(prediction)
prediction.columns = ["prediction"]
result = [x_test, y_test, prediction]
result = pd.concat([df for df in result], axis=1)
print(result)
print()
print("Accuracy:", accuracy(np.array(y_test).T[0], np.array(prediction).T[0], average=True))
print("Precision:", precision(np.array(y_test).T[0], np.array(prediction).T[0], average=True))
print("Recall:", recall(np.array(y_test).T[0], np.array(prediction).T[0], average=True))
print("F1 Score:", f1_score(np.array(y_test).T[0], np.array(prediction).T[0], average=True))


