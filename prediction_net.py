import tensorflow as tf 
import numpy as np

class PredictionNet:
	def __init__(self,past_prices,signal1,signal2,signal3,signal4,signal5):
		self.past_prices = past_prices
		self.signal1 = signal1
		self.signal2 = signal2
		self.signal3 = signal3
		self.signal4 = signal4
		self.signal5 = signal5

		# Init vars
		self.lstm = tf.contrib.rnn.BasicLSTMCell(16)
		self.w1 = tf.get_variable('w1',[16,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.b1 = tf.get_variable('b1',[32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.w2 = tf.get_variable('w2',[32,102],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.b1 = tf.get_variable('b1',[102],initializer=tf.truncated_normal_initializer(stddev=0.1))

	def forward(self,price_series,signals):
		batch_size, seq_len = self._shape(price_series)

		inital_state = self.lstm.zero_state(batch_size,tf.float32)
		outputs, hidden_state = tf.nn.dynamic_rnn(self.lstm,price_series,initial_state=inital_state,dtype=tf.float32,scope='lstm')

		layer = tf.concat([hidden_state,signals],1)
		layer = tf.nn.relu(tf.matmul(layer,self.w1) + self.b1)
		return tf.matmul(layer,self.w2) + self.b2 # softmax this

	def loss(self,logits,labels):
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))
		return loss

	def accuracy(self,logits,labels):
		n_correct= tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
		return tf.reduce_mean(tf.cast(n_correct, tf.float32))


	def _shape(self,tensor):
		size_0, size_1 = tensor.get_shape()
		arr = np.array([size_0,size_1], dtype=np.int32)
		return arr[0], arr[1]