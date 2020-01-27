# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# dataset = pd.read_csv('binary/combined1.csv')
#
# df1 = dataset.copy()
#
# df1_y = df1['marker']
# df1 = dataset.loc[:, dataset.columns != 'marker']
#
# df1 = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]
#
#
# def normalize(df):
#     result = df.copy()
#     for feature_name in df.columns:
#         max_value = df[feature_name].max()
#         min_value = df[feature_name].min()
#         result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
#     return result
#
#
# df1 = normalize(df1)
# df1 = df1.join(df1_y)
#
# dataset = df1.copy()
#
#
#
#
# train_set_percentage = 0.5
# fraud_series = dataset[dataset['marker'] == 1]
# idx = fraud_series.index.values
# np.random.shuffle(idx)
# fraud_series.drop(idx[:int(idx.shape[0]*train_set_percentage)], inplace=True)
# dataset.drop(fraud_series.index.values, inplace=True)
#
# normal_series = dataset[dataset['marker'] == 0]
# idx = normal_series.index.values
# np.random.shuffle(idx)
# normal_series.drop(idx[fraud_series.shape[0]:], inplace=True)
# dataset.drop(normal_series.index.values, inplace=True)
#
# new_dataset = pd.concat([normal_series, fraud_series])
# new_dataset.reset_index(inplace=True, drop=True)
# y = new_dataset['marker'].values.reshape(-1, 1)
# new_dataset.drop(['marker'], axis=1, inplace=True)
# X = new_dataset
#
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import recall_score
# from sklearn.metrics import confusion_matrix
# # Attributes that will be used by the gridsearchCV algorithm
# attr={'C': [0.1, 1, 2, 5, 10, 25, 50, 100],
#       'gamma': [1e-1, 1e-2, 1e-3]
#      }
#
# X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.3, random_state=10)
#
# model = SVC()
# classif = GridSearchCV(model, attr, cv=5)
# classif.fit(X_train, y_train)
# y_pred = classif.predict(X_test)
# print('Accuracy: ',accuracy_score(y_pred, y_test))
#
# y_all = dataset['Class'].values.reshape(-1, 1)
# dataset.drop(['Class'], axis=1, inplace=True)
# X_all = dataset
# y_pred_all = classif.predict(X_all)
# print(confusion_matrix(y_all, y_pred_all))
#
# print(recall_score(y_all, y_pred_all))


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, auc, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

data = pd.read_csv("binary/data1_mod.csv")

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

No_of_frauds= len(data[data["marker"]==1])
No_of_normals = len(data[data["marker"]==0])
total= No_of_frauds + No_of_normals
Fraud_percent= (No_of_frauds / total)*100
Normal_percent= (No_of_normals / total)*100

#Resampleing the dataset

y=data['marker']
X=data.drop(['marker'], axis=1)


# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12)


# Applying SVM Algorithm

#Using the rbf kernel to build the initail model.
classifier= svm.SVC(C= 1, kernel= 'linear', random_state= 0)

#Fit into Model
classifier.fit(X_train, y_train)

#Predict the class using X_test
y_pred = classifier.predict(X_test)

con_mat = confusion_matrix(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
cls_report = classification_report(y_test, y_pred)


def confus_matrix(CM):
    fig, ax = plot_confusion_matrix(conf_mat= CM)
    plt.title("The Confusion Matrix of full dataset using best_parameters")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
    print("The accuracy is "+str((CM[1,1]+CM[0,0])/(CM[0,0] + CM[0,1]+CM[1,0] + CM[1,1])*100) + " %")
    print("The recall from the confusion matrix is "+ str(CM[1,1]/(CM[1,0] + CM[1,1])*100) +" %")


confus_matrix(con_mat)


precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8,8))
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()