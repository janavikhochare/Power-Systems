import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
#%matplotlib inline

data = pd.read_csv("binary/combined.csv")

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
print(data.head())
count_classes = pd.value_counts(data['marker'], sort = True).sort_index()

from sklearn.preprocessing import StandardScaler

#data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

# hour = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)
# data['hour'] = StandardScaler().fit_transform(hour.reshape(-1, 1))

#data= data.drop(['Time','Amount'],axis=1)


class Autoencoder(object):

    def __init__(self, n_hidden_1, n_hidden_2, n_input, learning_rate):
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_input = n_input

        self.learning_rate = learning_rate

        self.weights, self.biases = self._initialize_weights()

        self.x = tf.placeholder("float", [None, self.n_input])

        self.encoder_op = self.encoder(self.x)
        self.decoder_op = self.decoder(self.encoder_op)

        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
        }

        return weights, biases

    def encoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    def decoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        # layer_1 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.add(tf.matmul(X, self.weights['decoder_h1']),
        #                                self.biases['decoder_b1'])))
        # layer_2 = tf.nn.relu(tf.layers.batch_normalization(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
        #                                self.biases['decoder_b2'])))
        # x_norm = tf.layers.batch_normalization(x, training=x)
        return layer_2

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.sess.run(self.encoder_op, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoder_op, feed_dict={self.x: X})
print("============================================================")
from sklearn.model_selection import train_test_split
good_data = data[data['marker'] == 0]
#print(good_data.shape)
bad_data = data[data['marker'] == 1]
#print(bad_data.shape)
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

X_train = X_train[X_train['marker']==0]
#print(X_train.shape)
X_train = X_train.drop(['marker'], axis=1)

y_test = X_test['marker']
X_test = X_test.drop(['marker'], axis=1)

X_train = X_train.values
X_test = X_test.values

X_good = good_data.ix[:, good_data.columns != 'marker']
y_good = good_data.ix[:, good_data.columns == 'marker']

X_bad = bad_data.ix[:, bad_data.columns != 'marker']
y_bad = bad_data.ix[:, bad_data.columns == 'marker']

model = Autoencoder(n_hidden_1=8, n_hidden_2=4, n_input=X_train.shape[1], learning_rate = 10e-3)

training_epochs = 100
batch_size = 128
display_step = 10
record_step = 10

total_batch = int(X_train.shape[0]/batch_size)

print(total_batch)

cost_summary = []

for epoch in range(training_epochs):
    cost = None
    for i in range(total_batch):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = X_train[batch_start:batch_end, :]
        
        cost = model.partial_fit(batch)
    
    if epoch % display_step == 0 or epoch % record_step == 0:
        total_cost = model.calc_total_cost(X_train)
        
        if epoch % record_step == 0:
            cost_summary.append({'epoch': epoch+1, 'cost': total_cost})
        
        if epoch % display_step == 0:
            print("Epoch:{}, cost={:.9f}".format(epoch+1, total_cost))
encode_decode = None
total_batch = int(X_test.shape[0]/batch_size) + 1


for i in range(total_batch):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch = X_test[batch_start:batch_end, :]
    batch_res = model.reconstruct(batch)
    if encode_decode is None:
        encode_decode = batch_res
    else:
        encode_decode = np.vstack((encode_decode, batch_res))

def get_df(orig, ed, _y):
    rmse = np.mean(np.power(orig - ed, 2), axis=1)
    return pd.DataFrame({'rmse': rmse, 'target': _y})
df = get_df(X_test, encode_decode, y_test)

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_auc_score, 
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support) 

import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        1

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


y_pred = [1 if p > 2 else 0 for p in df.rmse.values]
cnf_matrix = confusion_matrix(df.target, y_pred)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# x_norm = tf.layers.batch_normalization(x, training=x)
