import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
#import mpld3 as mpl
import itertools
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import sklearn.metrics
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)
from subprocess import check_output

data = pd.read_csv("binary/data1_mod.csv", header=0)
# data = pd.read_csv("binary/combined.csv")

df1 = data.copy()

df1_y = df1['marker']
df1 = data.loc[:, data.columns != 'marker']

df1 = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df1 = normalize(df1)
df1 = df1.join(df1_y)

data = df1.copy()
df = data.copy()

count_classes = pd.value_counts(df['marker'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

traindf, testdf = train_test_split(df, test_size=0.3)


def classification_model(model, data, predictors, outcome):
    scores = cross_val_score(model, data[predictors], data[outcome], cv=5, scoring='accuracy')
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


features = df.columns[0:-1]
outcome = 'marker'

classification_model(LogisticRegression(penalty='l2'), traindf, features, outcome)

model = LogisticRegression(penalty='l2')
model.fit(traindf[features], traindf[outcome])

predictions = model.predict(testdf[features])

print(sklearn.metrics.confusion_matrix(testdf[outcome], predictions, labels=None, sample_weight=None))

accuracy = metrics.accuracy_score(predictions, testdf[outcome])
print(accuracy)