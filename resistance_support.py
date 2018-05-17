from peak_detector import PeakDetector 
import matplotlib.pyplot as plt 
import datetime as dt
plt.style.use('ggplot')

class ResistanceSupport:
	def __init__(self,prices,dates,verbose=True):
		self.prices = prices
		self.dates = dates
		self.verbose = verbose

		self.resistance = {}
		self.support = {}


	def get_peaks(self,bandwidth,scan=2):
		self.detector = PeakDetector(self.prices,self.dates,bandwidth,self.verbose)
		self.detector.pick_peaks(scan)


	def get_resistance(self,same_mult=1.02):
		tops = self.detector.get_adjusted_tops()
		self.resistance[self.detector.bandwidth] = {}

		for i in range(len(tops)):
			top = tops[i]
			top_not_in_key = True

			for top_key in self.resistance[self.detector.bandwidth].keys():
				if (top <= top_key*same_mult) and (top >= top_key/same_mult):
					top_not_in_key = False

					if top > top_key:
						self.resistance[self.detector.bandwidth][top] = self.resistance[self.detector.bandwidth].pop(top_key)
						self.resistance[self.detector.bandwidth][top] += [self.dates[i]]

					else:
						self.resistance[self.detector.bandwidth][top_key] += [self.dates[i]]

			if top_not_in_key:
				self.resistance[self.detector.bandwidth][top] = [self.dates[i]]


	def get_support(self,same_mult=1.02):
		bots = self.detector.get_adjusted_bottoms()
		self.support[self.detector.bandwidth] = {}

		for i in range(len(bots)):
			bot = bots[i]
			bot_not_in_key = True

			for bot_key in self.support[self.detector.bandwidth].keys():
				if (bot <= bot_key*same_mult) and (bot >= bot_key/same_mult):
					bot_not_in_key = False

					if bot < bot_key:
						self.support[self.detector.bandwidth][bot] = self.support[self.detector.bandwidth].pop(bot_key)
						self.support[self.detector.bandwidth][bot] += [self.dates[i]]

					else:
						self.support[self.detector.bandwidth][bot_key] += [self.dates[i]]

			if bot_not_in_key:
				self.support[self.detector.bandwidth][bot] = [self.dates[i]]


	def plot_levels(self,date1,date2):
		## Organize dates and index
		while date1 not in self.dates:
			date1 += dt.timedelta(1)

		while date2 not in self.dates:
			date2 -= dt.timedelta(1)

		i1 = self.dates.index(date1)
		i2 = self.dates.index(date2)

		## Get levels within that range
		support_levels = []
		resistance_levels = []

		for price,dates in self.support[self.detector.bandwidth].items():
			date_in_range = False
			for date in dates:
				if (date >= date1) and (date <= date2):
					date_in_range = True

			if date_in_range:
				support_levels.append(price)

		for price,dates in self.resistance[self.detector.bandwidth].items():
			date_in_range = False
			for date in dates:
				if (date >= date1) and (date <= date2):
					date_in_range = True

			if date_in_range:
				resistance_levels.append(price)

		date_range = self.dates[i1:i2]
		N = len(date_range)

		resistance_plot = []
		for price in resistance_levels:
			price_lst = [price]*N
			resistance_plot.append(date_range)
			resistance_plot.append(price_lst)
			resistance_plot.append('r--')

		support_plot = []
		for price in support_levels:
			price_lst = [price]*N
			support_plot.append(date_range)
			support_plot.append(price_lst)
			support_plot.append('g--')

		levels_plot = support_plot + resistance_plot

		## plot
		plt.ylim(min(self.prices[i1:i2])-2,max(self.prices[i1:i2])+2)
		plt.title('Levels of support and resistance - bandwidth: {0} \n {1} to {2}'.format(self.detector.bandwidth,date1,date2))
		plt.xlabel('Dates')
		plt.ylabel('Closing Prices')

		plt.plot(date_range,self.detector.smoothed_prices[i1:i2],'b--',
				 date_range,self.prices[i1:i2],'k',*levels_plot)
		plt.gcf().autofmt_xdate()
		plt.show()



