import tensorflow as tf 
import numpy as np

class Discriminator:
	def __init__(self,hidden_size,max_seq_length,dropout=0.8,reuse=False):
		self.hidden_size = hidden_size
		self.max_seq_length = max_seq_length
		self.dropout = dropout

		if reuse:
			tf.get_variable_scope().reuse_variables()

		# Define trainable variables
		with tf.variable_scope("d_forward"):
			self.gru_fw = tf.contrib.rnn.GRUCell(self.hidden_size,reuse=reuse)
		with tf.variable_scope("d_backward"):
			self.gru_bw = tf.contrib.rnn.GRUCell(self.hidden_size,reuse=reuse)

		self.d_w1 = tf.get_variable('d_w1',[2*self.hidden_size,self.hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b1 = tf.get_variable('d_b1',[self.hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.1))

		self.d_w2 = tf.get_variable('d_w2',[self.hidden_size,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b2 = tf.get_variable('d_b2',[1],initializer=tf.truncated_normal_initializer(stddev=0.1))


	def forward(self,inp,hidden_state):
		# inp dim : batch_size * seq_len
		batch_size, seq_len = self._shape(inp)
		inp = tf.reshape(inp,shape=[batch_size,seq_len,1]) # : batch_size * seq_len * 1

		## Bi-directional GRU
		_, states = tf.nn.bidirectional_dynamic_rnn(
			self.gru_fw,
			self.gru_bw,
			inp,
			initial_state_fw=hidden_state,
			initial_state_bw=hidden_state,
			dtype=tf.float32,
			scope='d_bi_gru') 

		## Hidden layer
		state = tf.concat([states[0],states[1]],1) # : batch_size * (2*hidden_size)
		hidden_layer = tf.nn.dropout(tf.tanh(tf.matmul(state,self.d_w1) + self.d_b1),self.dropout) # batch_size * hidden)size

		## Output layer
		out = tf.sigmoid(tf.matmul(hidden_layer,self.d_w2) + self.d_b2) # batch_size * 1
		return tf.reshape(out,shape=[batch_size]) # batch_size


	def batch_classify(self,inp):
		num_samples, seq_len = self._shape(inp)
		hidden_state = self.gru_bw.zero_state(num_samples, tf.float32)
		return self.forward(inp,hidden_state) # num_samples

	def _shape(self,tensor):
		size_0, size_1 = tensor.get_shape()
		arr = np.array([size_0,size_1], dtype=np.int32)
		return arr[0], arr[1]


	# def batch_bce_loss(self,inp,targets):
	# 	out = self.batch_classify(inp)
	# 	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=targets))
	# 	return loss














