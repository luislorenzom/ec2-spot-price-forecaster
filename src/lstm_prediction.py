from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as md

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy

import math

# ------------------------------------------------------
# LOAD DATA
#  -----------------------------------------------------
# Define what dataset we're going to use
DATASET='../datasets/eu-west-1.csv'
INSTANCE_TYPE='m4.4xlarge'
OS='Linux/UNIX'
REGION='eu-west-1a'

# Load dataset
df = pd.read_csv(DATASET, names=["time", "instance_type", "os", "region", "price"])

# Filtering data
filter_1 = df["instance_type"]==INSTANCE_TYPE
filter_2 = df["os"]==OS
filter_3 = df["region"]==REGION
df.where(filter_1 & filter_2 & filter_3, inplace = True)
data = df.dropna()
time = pd.to_datetime(data.time)

# Plotting
x = list(time)
y = list(data['price'])

#plt.plot(x,y)
#plt.show()


# ------------------------------------------------------
# PREPARE DATA
#  -----------------------------------------------------
# Prepare the dataset
dataset = data['price'].values
dataset = dataset.astype('float32')

# Normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)

# Split into training and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)

# Reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
trainY = numpy.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# ------------------------------------------------------
# DEFINE MODEL ARCHITECTURE
#  -----------------------------------------------------
# Define model
model = Sequential()

# Define layers
model.add(LSTM(32, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Model compiling
model.compile(loss='mean_squared_error', optimizer='nadam')


# ------------------------------------------------------
# MODEL TRAINING
#  -----------------------------------------------------
model.fit(trainX, trainY, epochs=130, batch_size=1, verbose=2)


# ------------------------------------------------------
# PREDICTIONS AND GET METRICS
#  -----------------------------------------------------
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))