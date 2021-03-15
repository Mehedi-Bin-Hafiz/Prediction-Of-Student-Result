import pandas as pd
import warnings
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
#database
import matplotlib.pyplot as plt
from keras import backend as K
def r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

MainDatabase = pd.read_excel("Database.xlsx").iloc[:1500]
y = MainDatabase.iloc[ : , 5:6].values
from sklearn.preprocessing import MinMaxScaler
sc =MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(y)
# print(training_set_scaled)

x_train = []
y_train = []

#timestamp 60 means I will create 60 features.

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

x_train, y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))
# print(x_train)
# #input shape must be (batch_size, timesteps, input_dim)
#
from keras.models import Sequential
from keras.layers import Dense , LSTM, Dropout
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss = 'mean_squared_error',metrics=[r2_score])

history = model.fit(x_train,y_train,epochs=10,batch_size=16)


history_dict = history.history
print(history_dict.keys())


loss_values = history_dict['loss']
epochs = range(1,len(loss_values)+1)
plt.plot(epochs, loss_values, label = "loss")
plt.xlabel('Epochs')
plt.ylabel('Scores')
plt.legend()
plt.show()

R2Socre = history_dict['r2_score']
epochs = range(1,len(R2Socre)+1)
plt.plot(epochs, R2Socre,label = 'R2 score')
plt.xlabel('Epochs')
plt.ylabel('Scores')
plt.legend()
plt.show()