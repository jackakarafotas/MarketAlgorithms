import tensorflow as tf 
import numpy as np

class Generator:
	def __init__(self,hidden_size,max_seq_length,reuse=False):
		self.hidden_size = hidden_size
		self.max_seq_length = max_seq_length

		if reuse:
			tf.get_variable_scope().reuse_variables()

		# Define trainable variables
		with tf.variable_scope("g_gru"):
			self.gru = tf.contrib.rnn.GRUCell(self.hidden_size,reuse=reuse)

		self.g_w1 = tf.get_variable('g_w1',[self.hidden_size,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.g_b1 = tf.get_variable('g_b1',[1],initializer=tf.truncated_normal_initializer(stddev=0.1))


	def forward(self,inp,hidden_state):
		"""
		Inputs sequence one return at a time and runs it through a GRU

		INPUT:
		- inp : inp return of size [batch_size]
		- hidden_state : hidden state of pervious input or initial state of size [batch_size,hidden_size]

		OUTPUT:
		- output : prediction of next return made by generator of size [batch_size]
		- state : hidden state output by GRU to be used in next prediction of size [batch_size,hidden_size]
		"""

		# inp 											: batch_size 
		x = tf.reshape(inp,shape=[-1,1,1]) # 			: batch_size * 1 * 1

		# GRU Layer
		outputs, state = tf.nn.dynamic_rnn(self.gru,x,initial_state=hidden_state,dtype=tf.float32,scope='g_gru') # : batch_size * 1 * hidden_size

		# Output layer
		output = tf.tensordot(outputs,self.g_w1,axes=[[2],[0]]) + self.g_b1 # : batch_size * 1 * 1
		output = tf.reshape(output,shape=[-1]) # : batch_size
		output = tf.tanh(output)

		return output, state


	def sample(self,num_samples,init_return=0):
		"""
		Samples num_samples from the network

		INPUT:
		- num_samples : number of sampels wanted 
		- init_return : initial return to initialize the generation process
		"""
		hidden_state = self.gru.zero_state(num_samples, tf.float32)
		inp = tf.constant(init_return,shape=[num_samples],dtype=tf.float32) # : num_samples

		for i in range(self.max_seq_length):
			inp, hidden_state = self.forward(inp,hidden_state) # out : num_samples

			if i == 0:
				samples = tf.reshape(inp,shape=[num_samples,1]) # init samples : num_samples * 1
			else:
				samples = tf.concat([samples,tf.reshape(inp,shape=[num_samples,1])],1) # add columns -> num_samples * i -> num_samples * max_seq_length

		return samples


	def batch_mse_loss(self,inp,target):
		"""
		Returns the batch mse loss for predicting a target sequence. Used in MLE training.

		Inputs:
		- inp: batch_size * seq_len - should be target with init return prepended 
		- target: batch_size * seq_len
		"""

		batch_size, seq_len = self._shape(inp)
		hidden_state = self.gru.zero_state(batch_size, tf.float32)
 
		loss = 0
		for i in range(seq_len):
			x = inp[:,i]
			y = target[:,i]

			out, hidden_state = self.forward(x,hidden_state) # : batch_size
			loss += tf.losses.mean_squared_error(labels=y,predictions=out)

		return loss


	def batch_pg_loss(self,inp,target,rewards):
		"""
		Returns a pseudo-loss that gives corresponding policy gradients

		INPUTS:
		- inp: batch_size * seq_len - should be target with init return prepended 
		- target: batch_size * seq_len
		- reward: batch_size (discriminator reward for each return series)
		"""

		batch_size, seq_len = self._shape(inp)
		hidden_state = self.gru.zero_state(batch_size, tf.float32)

		loss = 0
		for i in range(seq_len):
			x = tf.reshape(inp[:,i],shape=[batch_size])		# : batch_size
			y = tf.reshape(target[:,i],shape=[batch_size])	# : batch_size

			out, hidden_state = self.forward(x,hidden_state) # : batch_size

			loss += tf.reduce_mean(tf.squared_difference(y,out)*rewards)

		return loss

	def _shape(self,tensor):
		size_0, size_1 = tensor.get_shape()
		arr = np.array([size_0,size_1], dtype=np.int32)
		return arr[0], arr[1]








