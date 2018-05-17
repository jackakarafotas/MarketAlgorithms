import tensorflow as tf
from base_model import BaseModel 

class BasicGan(BaseModel):
	def generator(self,batch_size,reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		self.z = tf.truncated_normal([batch_size,self.z_dim],mean=0,stddev=0.01,name='z')

		# layer 1
		self.g_w1 = tf.get_variable('g_w1',[self.z_dim,2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g_b1 = tf.get_variable('g_b1',[2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g1 = tf.matmul(self.z,self.g_w1) + self.g_b1
		self.g1 = tf.contrib.layers.batch_norm(self.g1,epsilon=1e-5,scope='bn1')
		self.g1 = tf.tanh(self.g1)
		self.g1 = tf.nn.dropout(self.g1, 0.9)

		# layer 2
		self.g_w2 = tf.get_variable('g_w2',[2048,1024],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g_b2 = tf.get_variable('g_b2',[1024],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g2 = tf.matmul(self.g1,self.g_w2) + self.g_b2
		self.g2 = tf.contrib.layers.batch_norm(self.g2,epsilon=1e-5,scope='bn2')
		self.g2 = tf.tanh(self.g2)
		self.g2 = tf.nn.dropout(self.g2, 0.9)

		# ouput layer
		self.g_w3 = tf.get_variable('g_w3',[1024,self.seq_length],initializer=tf.truncated_normal_initializer(stddev=0.01))
		self.g_b3 = tf.get_variable('g_b3',[self.seq_length],initializer=tf.truncated_normal_initializer(stddev=0.01))
		return tf.nn.tanh(tf.matmul(self.g2,self.g_w3) + self.g_b3)

	def discriminator(self,x,reuse=False):
		def conv2d(x_conv,W_conv):
			return tf.nn.conv2d(x_conv,W_conv,strides=[1,1,1,1],padding="SAME")

		def max_pool(x_conv,pool_size_conv):
			return tf.nn.max_pool(x_conv,ksize=[1,pool_size_conv,1,1],strides=[1,pool_size_conv,1,1],padding="SAME")

		if reuse:
			tf.get_variable_scope().reuse_variables()

		self.x_r = tf.reshape(x,shape=[-1,self.seq_length,1,1])

		# layer 1
		self.d_w1 = tf.get_variable('d_w1',[50,1,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b1 = tf.get_variable('d_b1',[32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		d_pool_size1 = 5
		self.d1_conv = tf.tanh(conv2d(self.x_r,self.d_w1) + self.d_b1)
		# split between positive and negative
		self.d1_pos = max_pool(self.d1_conv,d_pool_size1) # max_pool on positive
		self.d1_neg = max_pool(tf.negative(self.d1_conv),d_pool_size1) # max_pool on negative (turned into pos)

		d_pool_size2 = 2
		# layer 2 pos
		self.d_w2_pos = tf.get_variable('d_w2_pos',[5,1,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b2_pos = tf.get_variable('d_b2_pos',[64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d2_pos = max_pool(tf.nn.relu(conv2d(self.d1_pos,self.d_w2_pos)+ self.d_b2_pos),d_pool_size2)

		# layer 2 neg
		self.d_w2_neg = tf.get_variable('d_w2_neg',[5,1,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b2_neg = tf.get_variable('d_b2_neg',[64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d2_neg = max_pool(tf.nn.relu(conv2d(self.d1_neg,self.d_w2_neg)+ self.d_b2_neg),d_pool_size2)

		# combine
		self.d2 = tf.concat([self.d2_pos,self.d2_neg],1)

		# Hidden layers
		# layer 3
		in_dim = 2 * int(64 * (250/(d_pool_size1*d_pool_size2)))
		self.d_w3 = tf.get_variable('d_w3',[in_dim,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b3 = tf.get_variable('d_b3',[512],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d3 = tf.nn.leaky_relu(tf.matmul(tf.reshape(self.d2,[-1,in_dim]),self.d_w3) + self.d_b3)
		self.d3 = tf.nn.dropout(self.d3, 0.9)

		# layer 4
		self.d_w4 = tf.get_variable('d_w4',[512,256],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b4 = tf.get_variable('d_b4',[256],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d4 = tf.nn.leaky_relu(tf.matmul(self.d3,self.d_w4) + self.d_b4)
		self.d4 = tf.nn.dropout(self.d4, 0.9)

		# output layer
		self.d_w4 = tf.get_variable('d_w4',[256,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
		self.d_b4 = tf.get_variable('d_b4',[1],initializer=tf.truncated_normal_initializer(stddev=0.1))
		return tf.matmul(self.d3,self.d_w4) + self.d_b4

	def _get_name(self):
		self.model_name = 'BasicGan'
