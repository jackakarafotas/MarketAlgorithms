from data_loader import DataLoader
from peak_detector import PeakDetector  
# from resistance_support import ResistanceSupport 
# from kernel_regressor import KernelRegressor
import datetime as dt 
import matplotlib.pyplot as plt 
import sys
plt.style.use('ggplot')

## load data
date_start = dt.datetime(1998,1,1)
date_end = dt.datetime(2017,12,31)

data = DataLoader(date_start,date_end,verbose=False)
jpm = data.get_ticker_data('JPM') # from 1998 (first value) till 2017 (last)
jpm_closing = jpm['Close'].values.tolist()
jpm_dates = jpm.index.values.tolist()
jpm_dates = data.str_to_date(jpm_dates)

## Setup peak detector
# bandwidth = float(sys.argv[1])
# scan = int(sys.argv[2])
# detector = PeakDetector(jpm_closing,jpm_dates,bandwidth=bandwidth)
# detector.pick_peaks(scan=scan)
# date1 = dt.datetime(2002,1,1)
# date2 = dt.datetime(2003,1,1)
# detector.plot(date1,date2)

bandwidths = [5,10,15,20,25,30]
for bw in bandwidths:
	scan = int(bw)
	detector = PeakDetector(jpm_closing,jpm_dates,bandwidth=bw)
	detector.pick_peaks(scan=scan)
	detector.get_move_lengths()
	detector.plot_move_lengths_freq()




# levels = ResistanceSupport(jpm_closing,jpm_dates)
# levels.get_peaks(10,5)
# levels.get_resistance()
# levels.get_support()
# levels.plot_levels(date1,date2)