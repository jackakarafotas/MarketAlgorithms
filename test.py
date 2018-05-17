from data_loader import DataLoader 
from basic_gan import BasicGan 
from lstm_gan import LSTMGan
import datetime as dt 

## Constants
BATCH_SIZE = 10
Z_DIM = 512
LEARNING_RATE = 1E-4

## load data
date_start = dt.datetime(1998,1,1)
date_end = dt.datetime(2017,12,31)

data = DataLoader(date_start,date_end,verbose=False)
data.scrape_tickers()
data.get_data(data.tickers)
data.batch_returns()

## Get model
model = BasicGan(data.batched_returns,BATCH_SIZE,Z_DIM,LEARNING_RATE)
model.train()
model.generate(1)