# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 17:33:32 2018

@author: avalluru
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import tensorflow as tf
#import keras as k
#from keras.models import Sequential
#from keras.layers import Dense, Activation,LSTM """

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
pd.set_option("display.max_columns",101)
input_data=pd.read_csv("C:/Users/avalluru/Desktop/Analytics/Python projects/LMP_project1/LMPCleanData_Y1_8H.csv")
#show the numerical data in each column
#groups=[4,5,6,7]
groups=[7]
plt.figure(figsize=(15,12))
j=1
for station in input_data.Station_ID.unique():
    data=input_data.loc[input_data['Station_ID'] == station]
#data=input_data
#i=1
    for group in groups:
        plt.subplot(4,len(groups),j)
        plt.plot(data.values[:,group])
        plt.title(station+'_'+data.columns[group])
        j+=1;
plt.show()

#convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#get Month and day of week information
data['Date'] = pd.to_datetime(data['Date'])
data['day_of_week'] = data['Date'].dt.day_name()
data['Month'] = data['Date'].dt.month
data['Hour_str']='HR' + data['Hour'].astype(str)
#create dummy variables for hour,column drop and reorder
data1 = pd.concat([data, pd.get_dummies(data['Hour_str'])], axis=1);
data2=pd.concat([data1, pd.get_dummies(data['day_of_week'])], axis=1);
data3=data2.drop(['Unnamed: 0','Date','Hour','Station_ID','Hour_str','day_of_week','Month'],axis=1)
cols = list(data3)
cols.insert(0, cols.pop(cols.index('LMP')))
data3 = data3.loc[:, cols]

#get the LMP value
dataLMP=data3.loc[:, ['LMP']]
dataLMP = dataLMP.astype('float32')
#transform to supervised data frame
scalerLMP = MinMaxScaler(feature_range=(0, 1))
scaledLMP = scalerLMP.fit_transform(dataLMP)

#transform to supervised data frame
testReframed = series_to_supervised(scaledLMP, 24, 1,True)
testReframed=testReframed.reset_index()
testReframed=testReframed.drop(['index'],axis=1)

#drop the LMP column
dataWoLMP=data3.drop(['LMP'],axis=1)
dataWoLMP=dataWoLMP.iloc[24:,:]
dataWoLMP=dataWoLMP.reset_index()
dataWoLMP=dataWoLMP.drop(['index'],axis=1)
#Concatenate with the t-(t-24) data
preData = pd.concat([dataWoLMP, testReframed], axis=1, sort=False)
#Normolize data
cols_to_norm = ['mean_air','mean_hum','mean_wind_speed']
preData[cols_to_norm] = preData[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#split into train and test sets
values = preData.values
n_train_hours = 300 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
model = Sequential()
model.add(LSTM(72, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=500, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled=scaler.fit_transform(dataLMP)
reshape_test_y=np.reshape(test_y,(test_y.shape[0],1))
inv_yhat1 = scaler.inverse_transform(yhat)
inv_test_y=scaler.inverse_transform(reshape_test_y)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_yhat1, inv_test_y))
print('Test RMSE: %.3f' % rmse)
cv=rmse/np.mean(inv_test_y)
print('Test cv: %.3f' % cv)
plt.figure(figsize=(20,10))
plt.plot(inv_yhat1)
plt.plot(inv_test_y)
dataValue=data2.values
hoursData = dataValue[n_train_hours+24:, 2]
plt.figure(figsize=(20,10))
plt.plot(hoursData,(inv_yhat1-inv_test_y)/inv_test_y,'bo')
ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.axhline(y=0, xmin=0.0, xmax=1.0, color='r')