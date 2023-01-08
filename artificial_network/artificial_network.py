# Import libraries for data manipulation and analysis
import numpy as np
import pandas as pd
import scipy as sp

# Import libraries for visualization
import matplotlib.pyplot as plt
import plotly.express as px

# Import libraries for database management
import sqlalchemy

# Import library for stock data
import investpy

# Get stock data for Apple
stock_data = investpy.stocks.get_stock_historical_data(stock='AAPL', country='United States', from_date='01/01/2010', to_date='01/01/2020')

# Preprocess the data
stock_data = stock_data.dropna()  # Remove missing values
stock_data = stock_data.sort_values('Date')  # Sort the data by date
stock_data['Close'] = stock_data['Close'].astype(float)  # Convert 'Close' column to float

# Split the data into input (X) and output (y) variables
X = stock_data['Date'].values.reshape(-1, 1)  # Convert 'Date' column to numpy array and reshape to 2D array
y = stock_data['Close'].values  # Convert 'Close' column to numpy array

# Split the data into training and testing sets
num_train = int(len(X) * 0.8)
X_train, X_test = X[:num_train], X[num_train:]
y_train, y_test = y[:num_train], y[num_train:]

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=10)

# Make predictions on the testing data
y_pred = model.predict(X_test_scaled)

# Calculate root mean squared error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Visualize the results
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, label='Predicted', color='red')
plt.legend()
plt.show()
