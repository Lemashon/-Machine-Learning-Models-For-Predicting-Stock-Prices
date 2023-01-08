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
from sklearn.linear_model import LinearRegression

# Define the fitness function
def fitness_function(coefficients):
  model = LinearRegression(coefficients)
  model.fit(X_train_scaled, y_train)
  y_pred = model.predict(X_test_scaled)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  return rmse

# Define the genetic algorithm
def genetic_algorithm(population_size, num_generations):
  # Initialize the population with random coefficient values
  population = np.random.rand(population_size, X_train_scaled.shape[1])

  # Evaluate the fitness of each individual in the population
  fitness_values = [fitness_function(individual) for individual in population]

  # Iterate through the number of generations
  for i in range(num_generations):
    # Select the top performing individuals
    top_indices = np.argsort(fitness_values)[:int(population_size / 2)]
    top_individuals = population[top_indices]

    # Breed the top performing individuals to create a new generation
    offspring = []
    for j in range(int(population_size / 2)):
      parent_1 = top_individuals[j]
      parent_2 = top_individuals[np.random.randint(len(top_individuals))]
      child = (parent_1 + parent_2) / 2  # Simple crossover: take the average of the two parents
      offspring.append(child)

    # Add some randomness to the new generation by introducing mutation
    for j in range(len(offspring)):
      if np.random.rand() < 0.1:  # 10% chance of mutation
        offspring[j] = offspring[j] + np.random.normal(0, 0.1)

    # Replace the bottom performing individuals in the population with the new offspring
    population[top_indices] = offspring
    fitness_values[top_indices] = [fitness_function(individual) for individual in offspring]

  # Return the best performing individual
  best_index = np.argmin(fitness_values)
  return population[best_index]

# Run the genetic algorithm
best_coefficients = genetic_algorithm(population_size=100, num_generations=100)

# Train the model with the best performing coefficients
model = LinearRegression(best_coefficients)
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
