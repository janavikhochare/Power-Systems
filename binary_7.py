#########################
### OneR
#########################

import pandas as pd

class OneR(object):

    def __init__(self):
        self.ideal_variable = None
        self.max_accuracy = 0

    def fit(self, X, y):
        response = list()
        result = dict()

        dfx = pd.DataFrame(X)

        for i in dfx:
            result[str(i)] = dict()
            options_values = set(dfx[i])
            join_data = pd.DataFrame({"variable": dfx[i], "label": y})
            cross_table = pd.crosstab(join_data.variable, join_data.label)
            summary = cross_table.idxmax(axis=1)
            result[str(i)] = dict(summary)

            counts = 0

            for idx, row in join_data.iterrows():
                if row['label'] == result[str(i)][row['variable']]:
                    counts += 1

            accuracy = (counts / len(y))

            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.ideal_variable = i

            result_feature = {"variable": str(i), "accuracy": accuracy, "rules": result[str(i)]}
            response.append(result_feature)

        return response

    def predict(self, X=None):
        self_ideal_variable = self.ideal_variable + 1

    def __repr__(self):
        if self.ideal_variable != None:
            txt = "The best variable for your data is: " + str(self.ideal_variable)
        else:
            txt = "The best variable has not yet been found, try to execute the fit method previously"
        return txt


data = pd.read_csv('binary/data1.csv')
y_mush = data['marker']

x_mush = data.drop('marker', 1)

clf_mush = OneR()
results = clf_mush.fit(x_mush, y_mush)

print(clf_mush)

import numpy as np
from sklearn.model_selection import train_test_split

num = 10
clf_mush_cv = OneR()
accuracy_items = list()

for i in range(num):
    x_train, x_test, y_train, y_test = train_test_split(
        x_mush, y_mush,
        test_size=2,
        random_state=42)

    clf_mush_cv.fit(x_train, y_train)
    accuracy_items.append(clf_mush_cv.max_accuracy)

print(sum(accuracy_items) / num)