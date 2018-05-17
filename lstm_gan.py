import tensorflow as tf
from tensorflow.contrib import rnn
from base_model import BaseModel 

class LSTMGan(BaseModel):
	def generator(self,batch_size,reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		self.z = tf.truncated_normal([batch_size,self.z_dim],mean=0,stddev=0.1,name='z')
		self.z = tf.reshape(self.z,shape=[-1,self.z_dim,1])

		# LSTM Layer
		self.lstm = rnn.BasicLSTMCell(256)
		self.initalState = self.lstm.zero_state(batch_size,tf.float32)
		self.output, self.nextState = tf.nn.dynamic_rnn(self.lstm,self.z,initial_state=self.initalState)
		self.output = self.output[-1]

		# layer 1
		self.g_w1 = tf.get_variable('g_w1',[256,2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g_b1 = tf.get_variable('g_b1',[2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g1 = tf.matmul(self.output,self.g_w1) + self.g_b1
		self.g1 = tf.contrib.layers.batch_norm(self.g1,epsilon=1e-5,scope='bn1')
		self.g1 = tf.tanh(self.g1)
		self.g1 = tf.nn.dropout(self.g1, 0.8)

		# output layer
		self.g_w2 = tf.get_variable('g_w2',[2048,self.seq_length],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g_b2 = tf.get_variable('g_b2',[self.seq_length],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g2 = tf.matmul(self.g1,self.g_w2) + self.g_b2
		return tf.tanh(self.g2)

	def discriminator(self,x,reuse=False):
		def conv2d(x_conv,W_conv):
			return tf.nn.conv2d(x_conv,W_conv,strides=[1,1,1,1],padding="SAME")

		def avg_pool(x_conv,pool_size_conv):
			return tf.nn.avg_pool(x_conv,ksize=[1,pool_size_conv,1,1],strides=[1,pool_size_conv,1,1],padding="SAME")

		if reuse:
			tf.get_variable_scope().reuse_variables()

		self.x_r = tf.reshape(x,shape=[-1,self.seq_length,1,1])

		# layer 1
		self.d_w1 = tf.get_variable('d_w1',[5,1,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b1 = tf.get_variable('d_b1',[32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		d_pool_size1 = 2
		self.d1 = avg_pool(tf.nn.relu(conv2d(self.x_r,self.d_w1)+ self.d_b1),d_pool_size1)

		# layer 2
		self.d_w2 = tf.get_variable('d_w2',[5,1,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b2 = tf.get_variable('d_b2',[64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		d_pool_size2 = 5
		self.d2 = avg_pool(tf.nn.relu(conv2d(self.d1,self.d_w2)+ self.d_b2),d_pool_size2)

		# layer 3
		in_dim = int(64 * (250/(d_pool_size1*d_pool_size2)))
		self.d_w3 = tf.get_variable('d_w3',[in_dim,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b3 = tf.get_variable('d_b3',[1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d3 = tf.nn.relu(tf.matmul(tf.reshape(self.d2,[-1,in_dim]),self.d_w3) + self.d_b3)

		# output layer
		self.d_w4 = tf.get_variable('d_w4',[1024,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b4 = tf.get_variable('d_b4',[1],initializer=tf.truncated_normal_initializer(stddev=0.1))
		return tf.matmul(self.d3,self.d_w4) + self.d_b4

	def _get_name(self):
		self.model_name = 'LstmGan'