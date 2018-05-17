import bs4 as bs 
import pickle
import requests
import os
import pandas as pd 
import datetime as dt 
import pandas_datareader.data as data
from random import shuffle

LINK = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
NAME = 'sp500tickers'
DIRECTORY = "stock_dataframes"

class DataLoader:
	def __init__(self,date_start,date_end,data_source='yahoo',directory=DIRECTORY,verbose=True):
		self.start = date_start
		self.end = date_end
		self.data_source = data_source
		self.directory = directory 
		self.verbose = verbose


	def scrape_tickers(self,link=LINK,file_name=NAME,output=False):
		""" Scrapes ticker from wikipedia link """
		resp = requests.get(link)
		soup = bs.BeautifulSoup(resp.text,'lxml')
		table = soup.find('table',{'class':'wikitable sortable'})

		self.tickers = []
		for row in table.findAll('tr')[1:]:
			ticker = row.findAll('td')[0].text
			self.tickers.append(ticker)

		with open(file_name+'.pickle','wb') as f:
			pickle.dump(self.tickers,f)

		if output:
			return self.tickers


	def save_scraped_data(self,file_name=NAME):
		""" Gets and saves down data scraped from scrape_tickers"""
		if not hasattr(self,"tickers"):
			name = file_name+'.pickle'
			if os.path.isfile(name):
				with open(name,'rb') as f:
					self.tickers = pickle.load(f)
			else:
				ValueError("Get Tickers First")

		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		for ticker in self.tickers:
			try:
				if not os.path.exists(self.directory+'/{0}.csv'.format(ticker)):
					self._log(ticker)
					df = data.DataReader(ticker,self.data_source,self.start,self.end)
					df.to_csv(self.directory+'/{0}.csv'.format(ticker))
				else:
					self._log('Already Have:{0}'.format(ticker))
			except:
				self._log('Cannot obtain data for {0}'.format(ticker))

		self._log('DONE')


	def get_ticker_data(self,ticker,override=False):
		ticker = ticker.replace(".","-")
		def load_and_save():
			self._log(ticker)
			df = data.DataReader(ticker,self.data_source,self.start,self.end)
			df.to_csv(self.directory+'/{0}.csv'.format(ticker))
			return df

		try:
			if not override:
				if not os.path.exists(self.directory+'/{0}.csv'.format(ticker)):
					return load_and_save()
				else:
					self._log('Already Have: {0}'.format(ticker))
					df = pd.read_csv(self.directory+'/{0}.csv'.format(ticker),index_col=0)
					return df
			else:
				return load_and_save()
		except:
			self._log('Cannot obtain data for {0}'.format(ticker))


	def get_data(self,tickers):
		self._log("Loading Data...")
		self.closing_prices = []
		for ticker in tickers:
			try:
				self.closing_prices.append(self.get_ticker_data(ticker)['Close'].values.tolist())
			except:
				pass

		self._log("Done")


	def _differentiate(self):
		if not hasattr(self,'closing_prices'):
			ValueError('Get Data First')
		else:
			self.returns = []
			for prices in self.closing_prices:
				prices_future = prices[1:]
				prices_past = prices[:-1]
				self.returns.append([100*(f-p)/p for f,p in zip(prices_future,prices_past)])


	def batch_returns(self,jump=10,batch_size=250):
		if not hasattr(self,'closing_prices'):
			ValueError('Get Data First')
		else:
			self._differentiate() # get returns

			self.batched_returns = []
			for returns_lst in self.returns:
				for i in range(0,len(returns_lst)-batch_size,jump):
					self.batched_returns.append(returns_lst[i:i+batch_size])

			shuffle(self.batched_returns)

	def str_to_date(self,date_string_lst):
		return [self._str_to_date(date_string) for date_string in date_string_lst]

	def _str_to_date(self,date_string):
		year = int(date_string[:4])
		month = int(date_string[5:7])
		day = int(date_string[-2:])
		return dt.datetime(year,month,day)


	def _log(self,s):
		if self.verbose:
			print(s)




		