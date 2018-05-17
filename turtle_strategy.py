from data_loader import DataLoader
import datetime as dt 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

## load data
date_start = dt.datetime(1998,1,1)
date_end = dt.datetime(2017,12,31)

data = DataLoader(date_start,date_end,verbose=False)
stock = data.get_ticker_data('AAPL') # from 1998 (first value) till 2017 (last)

stock_closing = stock['Close'].values.tolist()
stock_high = stock['High'].values.tolist()
stock_low = stock['Low'].values.tolist()

stock_dates = stock.index.values.tolist()
stock_dates = data.str_to_date(stock_dates)


def get_period_peak(closing_prices, dates, current_date, max_min, n_weeks = 4):
	past_date = current_date - dt.timedelta(n_weeks*7)
	while past_date not in dates:
		past_date -= dt.timedelta(1)

	i_past = dates.index(past_date)
	i_current = dates.index(current_date)
	return max_min(closing_prices[i_past:i_current])

def get_trailing_n(daily_highs, daily_lows,trail = 15):
	return sum([high - low for high,low in zip(daily_highs[-trail:],daily_lows[-trail:])]) / trail

portfolio_size = 100000
orig_size = 100000
risk_allownace = 0.02

own = 0
price_asset_owned = 0
profit = 0
losing = 0
losing_mult = 1

enter_price = []
exit_price = []
start_delta = 80
dates = stock_dates[start_delta:]

for i, date in enumerate(dates):
	curr_price = stock_closing[i+start_delta]

	if own == 0:
		high = get_period_peak(stock_closing[:i+1+start_delta],stock_dates[:i+1+start_delta],date,max,4)
		low = get_period_peak(stock_closing[:i+1+start_delta],stock_dates[:i+1+start_delta],date,min,4)

		if curr_price > high:
			n = get_trailing_n(stock_high[:i+1+start_delta], stock_low[:i+1+start_delta],10)
			n_low = curr_price - (2*n)

			own = 1
			amt = int((losing_mult**losing) * (portfolio_size * risk_allownace) / (2*n))
			price_asset_owned = curr_price

			enter_price.append(-curr_price)
			exit_price.append(0)

		elif (curr_price < low):
			n = get_trailing_n(stock_high[:i+1+start_delta], stock_low[:i+1+start_delta],10)
			n_high = curr_price + (2*n)

			own = -1
			amt = int((losing_mult**losing) * (portfolio_size * risk_allownace) / (2*n))
			price_asset_owned = curr_price

			enter_price.append(curr_price)
			exit_price.append(0)

		else:
			enter_price.append(0)
			exit_price.append(0)

	elif (own > 0):
		low = get_period_peak(stock_closing[:i+1+start_delta],stock_dates[:i+1+start_delta],date,min,2)
		LT_high = get_period_peak(stock_closing[:i+1+start_delta],stock_dates[:i+1+start_delta],date,max,11)

		if (curr_price >= LT_high) and (own == 1):
			own += 1
			amt += int(2 * (losing_mult**losing) * ((portfolio_size * risk_allownace) / (2*n)))
			price_asset_owned = (price_asset_owned + 2*curr_price)  / 3

		elif ((curr_price < low) or (curr_price <= n_low)):
			exit_profit = amt * (curr_price - price_asset_owned)
			profit += exit_profit
			portfolio_size += exit_profit
			print(profit,':',exit_profit)

			if exit_profit >= 0:
				losing = 0
			else:
				losing += 1

			price_asset_owned = 0
			own = 0 
			enter_price.append(0)
			exit_price.append(curr_price)

		else:
			enter_price.append(0)
			exit_price.append(0)

	elif (own < 0):
		high = get_period_peak(stock_closing[:i+1+start_delta],stock_dates[:i+1+start_delta],date,max,2)
		LT_low = get_period_peak(stock_closing[:i+1+start_delta],stock_dates[:i+1+start_delta],date,min,11)

		if (curr_price <= LT_low)  and (own == -1):
			own += -1
			amt += int(2 * (losing_mult**losing) * ((portfolio_size * risk_allownace) / (2*n)))
			price_asset_owned = (price_asset_owned + 2*curr_price)  / 3

		elif ((curr_price > high) or (curr_price >= n_high)):
			exit_profit = -own * (price_asset_owned - curr_price)
			profit += exit_profit
			portfolio_size += exit_profit
			print(profit,':',exit_profit)

			if exit_profit >= 0:
				losing = 0
			else:
				losing += 1

			own = 0 
			price_asset_owned = 0
			enter_price.append(0)
			exit_price.append(-curr_price)
		else:
			enter_price.append(0)
			exit_price.append(0)

print(enter_price[:100])
print(exit_price[:100])

print(profit)
print('Strategy:',portfolio_size)
buy_and_hold = orig_size + (0.05*orig_size*(stock_closing[-1] - stock_closing[start_delta]))
print('Buy and Hold:',buy_and_hold)
print('Strategy:',(portfolio_size-orig_size)/orig_size)
print('Buy and Hold:',(buy_and_hold-orig_size)/orig_size)

