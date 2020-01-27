import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
SEED = 123  # used to help randomly select the data points
DATA_SPLIT_PCT = 0.2
rcParams['figure.figsize'] = 8, 6
LABELS = ["Natural", "Attack"]

# df = pd.read_csv("binary/combined.csv")
# print(df['R1-PA:Z'].mean())
# df = df.fillna(df.mean())
# print(df['R1-PA:Z'].mean())
# df1 = df.copy()

df = pd.read_csv("binary/data1_mod.csv")
# df = df.fillna(df.mean())
df1 = df.copy()
df1_y = df1['marker']
df1 = df.loc[:, df.columns != 'marker']
# print(df1['R1-PA:Z'].max())
# print(df1['R1-PA:Z'].min())

# df_min = np.nanmin(df1, axis=1)
# df_max = np.nanmax(df1, axis=1)

# for feature_name in df1.columns:
#     m = df.loc[df[feature_name] != np.nan, feature_name].mean()
#     df[feature_name].replace(np.nan, m, inplace=True)
#
#     m = df.loc[df[feature_name] != np.inf, feature_name].max()
#     df[feature_name].replace(np.inf, m, inplace=True)

df1 = df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]


# print(df1['R1-PA:Z'].max())
# print(df1['R1-PA:Z'].min())

# df1['data'].fillna(df1[['a', 'b']].min(axis=1), inplace=True)


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df1 = normalize(df1)
df1 = df1.join(df1_y)

df = df1.copy()
# print("df: ", df.columns[df.isna().any()].tolist())

sign = lambda x: (1, -1)[x < 0]

df_train, df_test = train_test_split(df, test_size=DATA_SPLIT_PCT, random_state=SEED)
df_train, df_valid = train_test_split(df_train, test_size=DATA_SPLIT_PCT, random_state=SEED)

df_train_0 = df_train.loc[df['marker'] == 0]
df_train_1 = df_train.loc[df['marker'] == 1]
df_train_0_x = df_train_0.drop(['marker'], axis=1)
df_train_1_x = df_train_1.drop(['marker'], axis=1)

df_valid_0 = df_valid.loc[df['marker'] == 0]
df_valid_1 = df_valid.loc[df['marker'] == 1]
df_valid_0_x = df_valid_0.drop(['marker'], axis=1)
df_valid_1_x = df_valid_1.drop(['marker'], axis=1)

df_test_0 = df_test.loc[df['marker'] == 0]
df_test_1 = df_test.loc[df['marker'] == 0]
df_test_0_x = df_test_0.drop(['marker'], axis=1)
df_test_1_x = df_test_1.drop(['marker'], axis=1)

scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['marker'], axis=1))
df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['marker'], axis=1))

nb_epoch = 100
batch_size = 128
input_dim = df_train_0_x_rescaled.shape[1]  # num of predictor variables,
encoding_dim = 32
hidden_dim = int(encoding_dim / 2)
learning_rate = 10e-8
adam = optimizers.Adam(lr=10e-8, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=0.1)
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder)

# autoencoder = load_model('autoencoder_classifier.h5')

autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=adam)

cp = ModelCheckpoint(filepath="autoencoder_binary_classifier.h5",
                     save_best_only=True,
                     verbose=0)

tb = TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
                          epochs=nb_epoch,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                          verbose=1,
                          callbacks=[cp, tb]).history

valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_valid['marker']})
# precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
# plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
# plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
# plt.title('Precision and recall for different threshold values')
# plt.xlabel('Threshold')
# plt.ylabel('Precision/Recall')
# plt.legend()
# plt.show()

# f1_array = []
# max_f1_array = 0
# precision_recall_pos = 0
# for i in range(len(recall_rt)):
#     num = 2 * recall_rt[i] * precision_rt[i]
#     den = recall_rt[i] + precision_rt[i]
#     temp_f1_array = float(num / den)
#     f1_array.append(temp_f1_array)
#     if temp_f1_array > max_f1_array:
#         max_f1_array = temp_f1_array
#         precision_recall_pos = i
# print(max_f1_array, precision_recall_pos)
# print(precision_rt[precision_recall_pos])
# print(recall_rt[precision_recall_pos])

test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
print("mse: ", mse)
error_df_test = pd.DataFrame({'Reconstruction_error': mse, 'True_class': df_test['marker']})
error_df_test = error_df_test.reset_index()
threshold_fixed = 0.22
groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Attack" if name == 1 else "Natural")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

pred_y = [1 if e > threshold_fixed else 0  for e in error_df_test.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df_test.True_class, pred_y)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate, )
plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
