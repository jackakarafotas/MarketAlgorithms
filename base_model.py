import tensorflow as tf
import datetime as dt 
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class BaseModel:
	def __init__(self,data,batch_size,z_dim,learning_rate,context=None,verbose=True,plot_progress=True):
		self.data = data
		self.batch_size = batch_size
		self.z_dim = z_dim
		self.learning_rate = learning_rate
		self.context = None
		self.verbose = verbose
		self.plot_progress = plot_progress

		if self.context != None:
			self.context_dim = len(self.context[0])
		else:
			self.context_dim = None

		self.seq_length = len(self.data[0])
		self.n_samples = len(self.data)
		self._get_name()

		self._build_graph()

	def generator(self,batch_size,reuse=False):
		pass

	def discriminator(self,x,reuse=False):
		pass

	def _get_name(self):
		pass

	def _build_graph(self):
		self.x = tf.placeholder(tf.float32,[None,self.seq_length],name='x')

		with tf.variable_scope(tf.get_variable_scope()):
			self.Gz = self.generator(self.batch_size)
			self.Dx = self.discriminator(self.x)
			self.Dg = self.discriminator(self.Gz,reuse=True)

		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg,labels=tf.ones_like(self.Dg)))
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dx,labels=tf.ones_like(self.Dx)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dg,labels=tf.zeros_like(self.Dg)))
		self.d_loss = self.d_loss_real + self.d_loss_fake

		self.tvars = tf.trainable_variables()
		self.d_vars = [var for var in self.tvars if 'd_' in var.name]
		self.g_vars = [var for var in self.tvars if 'g_' in var.name]

		with tf.variable_scope(tf.get_variable_scope()):
			self.d_trainer_real = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss_real,var_list=self.d_vars)
			self.d_trainer_fake = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss_fake,var_list=self.d_vars)
			self.g_trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss,var_list=self.g_vars)

		self.saver = tf.train.Saver()

	def train(self,save_model=True):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		gLoss = 0 
		dLoss_real, dLoss_fake = 1, 1
		d_real_count, d_fake_count, g_count = 0, 0, 0

		for i in range(0,self.n_samples,self.batch_size):
			real_batch = self.data[i:i+self.batch_size]

			## TRAIN
			if dLoss_fake > 0.6:
				# Train Discriminator on generated images
				_, dLoss_real, dLoss_fake, gLoss = self.sess.run([self.d_trainer_fake,self.d_loss_real,self.d_loss_fake,self.g_loss],{self.x:real_batch})
				d_fake_count += 1

			if gLoss > 0.5:
				# Train the generator
				_, dLoss_real, dLoss_fake, gLoss = self.sess.run([self.g_trainer,self.d_loss_real,self.d_loss_fake,self.g_loss],{self.x:real_batch})
				g_count += 1

			if dLoss_real > 0.45:
				# If the discriminator classifies real images as fake, train on real values
				_, dLoss_real, dLoss_fake, gLoss = self.sess.run([self.d_trainer_real,self.d_loss_real,self.d_loss_fake,self.g_loss],{self.x:real_batch})
				d_real_count += 1

			## Save / plot
			if i % (5000*self.batch_size) == 0:
				n = 5
				with tf.variable_scope(tf.get_variable_scope()):
					graphs = self.sess.run(self.generator(n,reuse=True))
					d_fake = self.sess.run(self.discriminator(self.x),{self.x:graphs})
					d_real = self.sess.run(self.discriminator(self.x),{self.x:real_batch[:n]})

				self._log("\nTRAINING STEP: "+str(int(i/self.batch_size))+" OUT OF "+str(int(self.n_samples/self.batch_size))+" AT "+str(dt.datetime.now()))
				for j in range(n):
					self._log("Discriminator Fake classification: "+str(sigmoid(d_fake[j][0])))
				self._log('')
				for j in range(n):
					self._log("Discriminator Real classification: "+str(sigmoid(d_real[j][0])))

				if self.plot_progress:
					self.plot(graphs[:n],train_index=i,real=False)
					self.plot(real_batch[:n],train_index=i,real=True)

			if (i % (5000*self.batch_size) == 0) and save_model:
				save_path = self.saver.save(self.sess,"models/pretrained_"+self.model_name+".ckpt",global_step = i)
				self._log("\nsaved to {0}".format(save_path))


	def generate(self,n=5):
		return self.sess.run(self.generator(n,reuse=True))

	def plot(self,samples,train_index = None, real = False):
		if real:
			real_or_fake = "Real"
			color = 'b'
		else:
			real_or_fake = "Fake"
			color = 'r'

		for i, sample in enumerate(samples):
			ts = self.returns_to_ts(sample)

			plt.ylim(-2,max(200,max(ts)))
			if train_index != None:
				plt.title('{0} - Graph {1} \nTraining Index: {2}'.format(real_or_fake,i+1,train_index))
			else:
				plt.title('{0} - Graph {1}'.format(real_or_fake,train_index))
			plt.xlabel('Time')
			plt.ylabel('Prices')

			plt.plot(ts,color=color)
			plt.show()


	def returns_to_ts(self,returns,init_price=100):
		prices = [init_price]
		for i, perc in enumerate(returns):
			prices.append(prices[i] + (prices[i]*perc/100))

		return prices


	def _log(self,s):
		if self.verbose:
			print(s)

