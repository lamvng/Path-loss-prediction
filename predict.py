from __future__ import absolute_import, division, print_function
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


# Import dataset, return a panda dataframe
def import_data():
    path = os.getcwd()
    os.chdir(path)
    df = pd.read_csv('dataset.csv', header = 0)
    df = df.dropna(inplace=False)
    df = df.sample(frac = 1).reset_index(drop = True) # Shuffle the dataset
    '''
    longitude = plt.subplot2grid((4, 2), (0, 0))
    latitude = plt.subplot2grid((4, 2), (0, 1))
    elevation = plt.subplot2grid((4, 2), (1, 0))
    altitude = plt.subplot2grid((4, 2), (1, 1))
    clutterheight = plt.subplot2grid((4, 2), (2, 0))
    distance = plt.subplot2grid((4, 2), (2, 1))
    loss = plt.subplot2grid((4, 2), (3, 0), colspan = 2)
    
    longitude = sns.distplot(df['longitude'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = longitude)
    sns.distplot(df['latitude'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = latitude)
    sns.distplot(df['elevation'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = elevation)
    sns.distplot(df['altitude'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = altitude)
    sns.distplot(df['clutterheight'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = clutterheight)
    sns.distplot(df['distance'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = distance)
    sns.distplot(df['loss'], hist=True, color = 'blue', hist_kws = {'edgecolor':'black'}, ax = loss)
    plt.tight_layout()
    plt.show()
    #sns.pairplot(df[['distance','clutterheight', 'loss']])
    '''
    return df


# Preprocess the dataframe
def preprocess(df):
    scaler = preprocessing.MinMaxScaler()
    y = df.iloc[:,-1] # x is Dataframe
    X = df.drop('loss', axis = 1)
    X = scaler.fit_transform(X)
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return (X_train, y_train), (X_test, y_test)


# Build the model
# Features: 6, label: 1
def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(32, activation = tf.nn.relu, input_shape = (6,)))
    model.add(layers.Dense(32, activation = tf.nn.relu))
    model.add(layers.Dense(1))
    optimizer_function = keras.optimizers.RMSprop(lr = 0.001)
    model.compile(loss = 'mean_absolute_error',\
        optimizer = optimizer_function,\
        metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model


# Build the linear regression model
# Only one neuron and one layer
def build_linear():
    model = keras.Sequential([layers.Dense(1, activation = 'linear', input_shape = (6,))])
    optimizer_function = keras.optimizers.SGD()
    model.compile(loss = 'mean_absolute_error',\
        optimizer = optimizer_function,\
        metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model


# Train and save model
def train(model, X_train, y_train):
    csv_logger = keras.callbacks.CSVLogger('training.log', separator = ',', append = False)
    history = model.fit(X_train, y_train, epochs = 1000, callbacks=[csv_logger])
    model.save('model.h5')
    return history


# Visualize the training progress
def plot_training(history, score):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.subplot2grid((1, 2), (0, 0))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], 'b',\
        label = 'Converged Mean Absolute Error: %.4f' %score[0])
    plt.legend()
    plt.subplot2grid((1, 2), (0, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['mean_squared_error'], 'r',\
        label = 'Converged Mean Squared Error: %.4f' %score[2])
    plt.legend()
    plt.show()


# Visualize the predicted distribution
def plot_predict(y_predict):
    plt.hist(y_predict, range = (120, 160))
    plt.title('Predicted Values Distribution')
    plt.show()


# Visualize the test set distribution
def plot_test(y_test):
    plt.hist(y_test, range = (120, 160))
    plt.title('Test Dataset Distribution')
    plt.show()


# Visualize and evaluate the model
def plot_error(y_test, y_predict):
    error = y_predict - y_test
    plt.hist(error)
    plt.title('Prediction Error Distribution')

    plt.show()
    return error


df = import_data()
(X_train, y_train), (X_test, y_test) = preprocess(df)
model = build_model()
model_linear = build_linear()

history = train(model, X_train, y_train)
score = model.evaluate(X_test, y_test) # MAE, MSE
y_predict = model.predict(X_test).flatten()
plot_training(history, score)
plot_predict(y_predict)
plot_test(y_test)
plot_error(y_test, y_predict)
# np.std(error)
# score[0]