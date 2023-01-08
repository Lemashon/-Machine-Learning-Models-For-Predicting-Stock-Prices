import yfinance as yf
import pandas as pd

tickers = ["VOO","VGK","VPL"] #Subtitute for the tickers you want
df =yf.download(tickers,  start = "2021-02-01" , end = "2021-02-04")

import yfinance as yf

tickers = ["VOO"]
df = yf.download(tickers,  start = "2021-02-01" , end = "2021-02-04") 
newdf = df.reset_index() 
newdf

print(newdf)