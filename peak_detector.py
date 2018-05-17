from kernel_regressor import KernelRegressor 
import datetime as dt 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

class PeakDetector:
	def __init__(self,prices,dates,bandwidth=None,verbose=True):
		self.prices = prices
		self.dates = dates
		self.verbose = verbose

		self.deltas = self._differentiate_dates(self.dates)
		self.model = KernelRegressor(self.prices,self.deltas,bandwidth)
		self.bandwidth = self.model.bandwidth

		self._smooth()
		self._get_gradients()  #0th val = grad from 0th val to 1st val


	def pick_peaks(self,scan=1):
		self._log('Picking Peaks...')
		self.peaks = [0 for delta in self.deltas]
		self.bottoms = [0 for delta in self.deltas]
		self.tops = [0 for delta in self.deltas]
		N = len(self.gradients)
		i = 1
		while i < N-1:
			grad1 = self.gradients[i-1]
			grad2 = self.gradients[i]

			if grad1 == 0: # only possible at start of ts
				i += 1
			else:
				while grad2 == 0:
					i += 1
					grad2 = self.gradients[i]

				is_peak = self._pick_peak(grad1,grad2)

				# scan surrounding points
				if is_peak == 1:
					left_index = max(i-scan,0)
					right_index = min(i+1+scan,N+1)
					local_prices = self.prices[left_index:right_index]

					high_index = self._local_arg_extremum(local_prices,i,scan,max)
					self.tops[high_index] = 1
					self.peaks[high_index] = 1

				elif is_peak == -1:
					left_index = max(i-scan,0)
					right_index = min(i+1+scan,N+1)
					local_prices = self.prices[left_index:right_index]

					low_index = self._local_arg_extremum(local_prices,i,scan,min)
					self.bottoms[low_index] = 1
					self.peaks[low_index] = 1

				i += 1
		self._log('Done!')


	def plot(self,date1,date2):

		while date1 not in self.dates:
			date1 += dt.timedelta(1)

		while date2 not in self.dates:
			date2 -= dt.timedelta(1)

		i1 = self.dates.index(date1)
		i2 = self.dates.index(date2)

		tops = self.get_adjusted_tops()
		bots = self.get_adjusted_bottoms()

		plt.ylim(min(self.prices[i1:i2])-2,max(self.prices[i1:i2])+2)
		plt.title('Peaks and Troughs - Bandwidth: {0} \n {1} to {2}'.format(self.bandwidth,date1,date2))
		plt.xlabel('Dates')
		plt.ylabel('Closing Prices')

		plt.plot(self.dates[i1:i2],self.smoothed_prices[i1:i2],'b--',
				 self.dates[i1:i2],self.prices[i1:i2],'k',
				 self.dates[i1:i2],tops[i1:i2],'go',
				 self.dates[i1:i2],bots[i1:i2],'ro')
		plt.gcf().autofmt_xdate()
		plt.show()


	def get_adjusted_bottoms(self):
		return [peak*p for peak,p in zip(self.bottoms,self.prices)]
	
	def get_adjusted_tops(self):
		return [peak*p for peak,p in zip(self.tops,self.prices)]

	def get_move_lengths(self):
		self.move_lengths = []
		date_old = self.dates[0]
		first = True

		for idx, peak in enumerate(self.peaks):
			if (peak == 1):
				if not first:
					date_diff = (self.dates[idx] - date_old).days
					self.move_lengths.append(date_diff)
				else:
					first = False
				date_old = self.dates[idx]

		self.move_lengths.sort()


	def plot_move_lengths_freq(self,low_perc=0.1,high_perc=0.9):
		n_moves = len(self.move_lengths)
		mean = sum(self.move_lengths) / n_moves

		low_idx = int(n_moves * low_perc)
		low = self.move_lengths[low_idx]
		median_idx = int(n_moves * 0.5)
		median = self.move_lengths[median_idx]
		high_idx = int(n_moves * high_perc)
		high = self.move_lengths[high_idx]


		plt.title('Move Lengths - Bandwidth {}'.format(self.bandwidth))
		plt.xlabel('Move Lengths (Days)')
		plt.ylabel('Frequencies')
		plt.hist(self.move_lengths,bins=20,color='b',alpha=0.7)
		plt.axvline(mean,color='r',linewidth=1.5,label='Mean: {}'.format(round(mean,2)))
		plt.axvline(median,color='k',linewidth=1.5,label='Median: {}'.format(median))
		plt.axvline(low,color='k',linestyle='--',linewidth=1,label='Low: {}'.format(low))
		plt.axvline(high,color='k',linestyle='--',linewidth=1,label='High: {}'.format(high))
		plt.legend(loc='upper right')
		plt.show()


	## Helpers
	def _differentiate_dates(self,date_lst):
		return [(date - date_lst[0]).days for date in date_lst]

	def _smooth(self):
		self.smoothed_prices = self.model.smooth()

	def _get_gradients(self):
		future = self.smoothed_prices[1:]
		past = self.smoothed_prices[:-1]
		self.gradients = [f-p for f,p in zip(future,past)]

	def _pick_peak(self,grad1,grad2):
		if (grad1 > 0) and (grad2 < 0):
			return 1
		elif (grad1 < 0) and (grad2 > 0):
			return -1
		else:
			return 0

	def _local_arg_extremum(self,price_slice,i,scan_size,extremum_func):
		local_extremum = extremum_func(price_slice)
		if price_slice[scan_size] == local_extremum:
			return i
		else:
			local_index = price_slice.index(local_extremum)
			return (i-scan_size)+local_index


	def _log(self,s):
		if self.verbose:
			print(s)
