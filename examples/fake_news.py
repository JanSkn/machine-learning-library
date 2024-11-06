# Example for classification

# allows import from different folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pandas as pd
import numpy as np
from pylearn import MultinomialNaiveBayes, accuracy, precision, recall, f1_score, train_test_split

data = pd.read_csv("data/fake_news.csv")
data = data[["title", "real"]]      # remove unneccessary columns

X = data["title"]
y = data["real"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
x_train, x_test, y_train, y_test = pd.Series(x_train), pd.Series(x_test), pd.Series(y_train), pd.Series(y_test)

nb = MultinomialNaiveBayes()
nb.fit(x_train, y_train)
x_test, y_test = x_test.reset_index(drop=True), y_test.reset_index(drop=True)           # reset index to make DataFrame concatenation possible
prediction = nb.predict(x_test)
prediction = pd.DataFrame(prediction)
prediction.columns = ["prediction"]
result = [x_test, y_test, prediction]
result = pd.concat([df for df in result], axis=1)
print(result)
print()
# due to the small size of vocabulary, training over the whole dataset increases the measurements
print("Accuracy:", accuracy(np.array(y_test).T[0], np.array(prediction).T[0], average=True))
print("Precision:", precision(np.array(y_test).T[0], np.array(prediction).T[0], average=True))
print("Recall:", recall(np.array(y_test).T[0], np.array(prediction).T[0], average=True))
print("F1 Score:", f1_score(np.array(y_test).T[0], np.array(prediction).T[0], average=True))