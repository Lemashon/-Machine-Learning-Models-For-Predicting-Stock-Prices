# Import the necessary libraries
import yfinance as yf
import datetime
import pandas as pd
# Get the current date and time
now = datetime.datetime.now()

# Get the stock data for Amazon in a 5 minute time frame
stock_data = yf.download("AMZN", start=now - datetime.timedelta(days=30), end=now, interval='1h')
new_stock_data= stock_data.reset_index()
updated_stock_data= new_stock_data.reset_index()
updated_stock_data = updated_stock_data.rename(columns={'level_0':'Date', 'index':'Time'})

stock_data.to_csv('stock_data.csv', index=False)



# Extract the date and time from the 'Datetime' column and create new 'Date' and 'Time' columns
#stock_data['Date'] = stock_data['time'].apply(lambda x: x.date())


# Reorder the columns so that 'Date' and 'Time' are the first two columns
#stock_data = stock_data[['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Save the data to a CSV file
#stock_data.to_csv('stock_data.csv', index=False)

print(updated_stock_data)

# Print the DataFrame

