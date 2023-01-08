import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Get stock data for Amazon
now = datetime.datetime.now()
start = now - datetime.timedelta(days=60)
interval = '1h'
stock_data = yf.download("AMZN", start=start, end=now, interval=interval)
new_stock_data = stock_data.reset_index()
updated_stock_data = new_stock_data.rename(columns={'index':'Date', 'Close':'Close'})

# Preprocess the data
updated_stock_data = updated_stock_data.dropna()
updated_stock_data['Close'] = updated_stock_data['Close'].astype(float)
updated_stock_data = updated_stock_data.sort_values('Date')

# Split the data into input (X) and output (y) variables
X = updated_stock_data['Date'].values.reshape(-1, 1)
Y = stock_data['Close'].values



# Split the data into training and testing sets
num_train = int(len(X) * 0.8)
X_train, X_test = X[:num_train], X[num_train:]
Y_train, Y_test = Y[:num_train], Y[num_train:]



#Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Define the model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1))

#Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

#Train the model
model.fit(X_train_scaled, Y_train, epochs=5, validation_split=0.2)

#Make predictions on the testing data
Y_pred = model.predict(X_test_scaled)

#Calculate the accuracy of the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support
MSE = mean_squared_error(Y_test, Y_pred)
print(f'mean_squared_error: {MSE:.2f}')
MAE = mean_absolute_error(Y_test, Y_pred)
print(f'mean_absolute_error: {MAE:.2f}')
r2 = r2_score(Y_test, Y_pred)
print(f'r2_score: {r2:.2f}')

#Setting the threshold so as to accept a binary fire
THRESHOLD = 0.5

Y_test_binary = [1 if y >= THRESHOLD else 0 for y in Y_test]
Y_pred_binary = [1 if y >= THRESHOLD else 0 for y in Y_pred]

# Calculate the precision, recall, and f1-score for the binary classification model
precision, recall, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average='binary')


#Plot the results
plt.plot(['precision', 'recall', 'f1-score'], [precision, recall, f1], marker='.')
plt.show()

