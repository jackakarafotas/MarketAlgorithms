import math
import statistics as stats 

class KernelRegressor:
	def __init__(self,y,x=None,bandwidth=None,verbose=True):
		self.y = y
		self.verbose = verbose

		if x == None:
			self.x = list(range(len(y)))
		else:
			self.x = x

		if bandwidth == None:
			self.bandwidth = stats.stdev(self.y)*((4/3/len(self.y))**(1/5))
		else:
			self.bandwidth = bandwidth
		self._log('\nBANDWIDTH: {0}'.format(self.bandwidth))


	def smooth(self):
		self._log("Smoothing Curve...")

		self.smoothed_points = []
		for x_point in self.x:
			self.smoothed_points.append(self.find_point(self.x,self.y,x_point))

		self._log("Done!")
		return self.smoothed_points


	def CV(self,skip=2):
		N = len(self.y)
		square_diffs = []
		J = 0
		for i in range(0,N,skip):
			train_y = self._drop_point(self.y,i)
			test_y = self.y[i]

			train_x = self._drop_point(self.x,i)
			test_x = self.x[i]

			model_y = self.find_point(train_x,train_y,test_x)
			square_diffs.append((test_y-model_y)**2)
			J+= 1

		return sum(square_diffs) / J


	def find_point(self,x,y,x_point):
		kernels = []
		for x_val in x:
			kernels.append(self._kernel(x_point-x_val))

		top = 0
		bot = sum(kernels)
		for i in range(len(y)):
			top += kernels[i] * y[i]
		return top/bot


	def pick_bandwidth(self,bandwidths,skip=5):
		self._log("Old BandWidth: "+str(self.bandwidth))

		errors = []
		for h in bandwidths:
			self.bandwidth = h
			error = self.CV(skip)
			errors.append(error)
			self._log("BandWidth: "+str(self.bandwidth)+" - MSE: "+str(error))

		min_index = errors.index(min(errors))
		self.bandwidth = bandwidths[min_index]
		self._log("New BandWidth: "+str(self.bandwidth))


	def adjust_bandwidth(self,multiplier=0.3):
		self.bandwidth = self.bandwidth * multiplier
		self._log("New BandWidth: "+str(self.bandwidth))


	## Helpers
	def _kernel(self,value):
		bot = self.bandwidth * math.sqrt(2*math.pi)
		exp_top = -value**2
		exp_bot = 2*(self.bandwidth**2)
		return math.exp(exp_top/exp_bot) / bot

	def _drop_point(self,values,i):
		left = values[:i]
		right = values[i+1:]
		return left+right

	def _log(self,s):
		if self.verbose:
			print(s)