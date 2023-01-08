# Import library for SVM
from sklearn.svm import SVR

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
# Use the 'rbf' kernel and specify the regularization parameter C
model = SVR(kernel='rbf', C=1.0)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_scaled)

# Calculate root mean squared error (RMSE)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Visualize the results
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, label='Predicted', color='red')
plt.legend()
plt.show()
