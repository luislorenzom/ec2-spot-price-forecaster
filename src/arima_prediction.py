from pandas import DataFrame

import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

DATASET='../datasets/eu-west-1.csv'
INSTANCE_TYPE='m4.xlarge'
OS='Linux/UNIX'
REGION='eu-west-1c'

# Data load
df = pd.read_csv(DATASET, names=["time", "instance_type", "os", "region", "price"])

# Filtering
filter_1 = df["instance_type"]==INSTANCE_TYPE
filter_2 = df["os"]==OS
filter_3 = df["region"]==REGION

df.where(filter_1 & filter_2 & filter_3, inplace = True)
data = df.dropna()
#data["region"].astype("category")
time = pd.to_datetime(data.time)

# Plotting
x = list(time)
y = list(data['price'])

plt.plot(x,y)
#plt.savefig('foo.png')
plt.show()

# Fit model
