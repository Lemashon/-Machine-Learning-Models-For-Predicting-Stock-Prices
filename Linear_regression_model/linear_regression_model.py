import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os
import matplotlib.pyplot as plt


# Get stock data for Amazon
now = datetime.datetime.now()
start = now - datetime.timedelta(days=60)
interval = '1h'
stock_data = yf.download("AMZN", start=start, end=now, interval=interval)
new_stock_data = stock_data.reset_index()
updated_stock_data = new_stock_data.rename(columns={'index':'Date', 'Close':'Close'})

# Preprocess the data
updated_stock_data  = updated_stock_data.dropna()
updated_stock_data['Close'] = updated_stock_data['Close'].astype(float) 
updated_stock_data  = updated_stock_data.sort_values('Date')

#Store the data in a csv format
directory = 'Linear_regression_model'
filepath = os.path.join(directory, 'amazon_stock_data.csv')
updated_stock_data.to_csv("amazon_stock_data.csv", index=False)

# Split the data into input (X) and output (y) variables
X = updated_stock_data['Date'].values.reshape(-1, 1) 
y = stock_data['Close'].values

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
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_scaled)

# Calculate root mean squared error (RMSE)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Calculate R squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f'R squared: {r2:.2f}')

# Visualize the results
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, label='Predicted', color='red')
plt.legend()
plt.show()
