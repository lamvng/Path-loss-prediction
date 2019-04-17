
'''
The dataset is taken under Creative Commons — Attribution 4.0 International — CC BY 4.0,
from the following paper:

S.I. Popoola, A.A. Atayero, O.D. Arausi, V.O. Matthews
Path loss dataset for modeling radio wave propagation in smart campus environment
Data Brief., 17 (2018), pp. 1062-1073
https://doi.org/10.1016/j.dib.2018.02.026
https://www.sciencedirect.com/science/article/pii/S2352340918301422?via%3Dihub#bibliog0005
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

#TODO: Đang tới phần denorm


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
def normalize(df):
    scaler = preprocessing.MinMaxScaler()
    distance = df['distance'].values
    y = df.iloc[:,-1] # x is Dataframe
    X = df.drop('loss', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    return (X_train, y_train), (X_test, y_test)


def denormalize(norm_data):
    norm_data = norm_data.reshape(-1,1)
    scl = preprocessing.MinMaxScaler()
    denorm_data = scl.inverse_transform(norm_data)
    return denorm_data


# Build the model with tensorflow
# Features: 6, label: 1
def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(32, activation = tf.nn.relu, input_shape = (6,)))
    model.add(layers.Dense(32, activation = tf.nn.relu))
    optimizer_function = keras.optimizers.RMSprop(lr = 0.001)
    model.add(layers.Dense(1)) 
    model.compile(loss = 'mean_squared_error',\
        optimizer = optimizer_function,\
        metrics = ['mean_absolute_error', 'mean_squared_error'])
    return model


def train_model(data):
    (X_train, y_train), (X_test, y_test) = normalize(data)
    model = build_model()
    model.fit(X_train, y_train, epochs = 10)

# Train dữ liệu, in kết quả
df = import_data()
train_model(df)
#X_train, X_test, y_train, y_test = normalize(df)