from data_loader import DataLoader 
from seq_gan import SeqGan
import datetime as dt 

## Constants
START_RETURN = 0.0
BATCH_SIZE = 32
MLE_EPOCHS = 100 # easier = 5
ADV_EPOCHS = 50 # easier = 2
PT_STEPS = 50 # easier = 5
PT_EPOCHS = 3 # easier = 1
AT_STEPS = 5 # easier = 2
AT_EPOCHS = 3 # easier = 1
G_HIDDEN_SIZE = 32
D_HIDDEN_SIZE = 64

## load data
date_start = dt.datetime(1998,1,1)
date_end = dt.datetime(2017,12,31)

data = DataLoader(date_start,date_end,verbose=False)
data.scrape_tickers()
data.get_data(data.tickers)
data.batch_returns()

## Get model
model = SeqGan(data.batched_returns,START_RETURN,BATCH_SIZE,MLE_EPOCHS,ADV_EPOCHS,PT_STEPS,PT_EPOCHS,AT_STEPS,AT_EPOCHS,G_HIDDEN_SIZE,D_HIDDEN_SIZE)

# Training
#model.train_generator_mle()
#model.pretrain_discriminator()
model.adversarial_training()


