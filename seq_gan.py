import tensorflow as tf
import numpy as np
import sys
from math import ceil

from generator import Generator 
from discriminator import Discriminator

class SeqGan():
	def __init__(self,data,start_return,batch_size,mle_epochs,adv_epochs,pt_steps,pt_epochs,at_steps,at_epochs,g_hidden_size,d_hidden_size):
		self.data = data
		self.n_samples = len(data)
		self.seq_length = len(data[0])
		self.start_return = start_return
		self.batch_size = batch_size
		self.mle_epochs = mle_epochs
		self.adv_epochs = adv_epochs
		self.pt_steps = pt_steps	
		self.pt_epochs = pt_epochs
		self.at_steps = at_steps
		self.at_epochs = at_epochs

		self.generator = Generator(g_hidden_size,self.seq_length)
		self.discriminator = Discriminator(d_hidden_size,self.seq_length,dropout=0.8)

		# build graphs
		print('\nbuilding graphs...',end='')
		print(' mle...',end='')
		self._build_mle_graph()
		print(' pg...',end='')
		self._build_pg_graph()
		print(' dis...',end='')
		self._build_dis_graph()

		# trainers
		g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
		d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]

		self.mle_train = tf.train.AdamOptimizer(1e-2).minimize(self.mle_loss,var_list=g_vars)
		self.pg_train = tf.train.AdamOptimizer(1e-2).minimize(self.pg_loss,var_list=g_vars)
		self.dis_train = tf.train.AdagradOptimizer(1e-3).minimize(self.dis_loss,var_list=d_vars)

		# initialize
		print(' initializing variables...',end='')
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		print(' Done.')


	## GRAPHS
	def _build_mle_graph(self):
		self.mle_x = tf.placeholder(tf.float32,[self.batch_size,self.seq_length],name='mle_x')
		self.mle_y = tf.placeholder(tf.float32,[self.batch_size,self.seq_length],name='mle_y')
		self.mle_loss = self.generator.batch_mse_loss(self.mle_x,self.mle_y)

	def _build_pg_graph(self):
		self.pg_samples = self.generator.sample(self.batch_size*2)
		self.pg_inp, self.pg_target = self._prep_generator_batch_tf(self.pg_samples)
		self.pg_rewards = self.discriminator.batch_classify(self.pg_target)
		self.pg_loss = self.generator.batch_pg_loss(self.pg_inp, self.pg_target, self.pg_rewards)

	def _build_dis_graph(self):
		self.dis_x = tf.placeholder(tf.float32,[self.batch_size,self.seq_length],name='dis_x')
		self.dis_y = tf.placeholder(tf.float32,[self.batch_size],name='dis_y')

		self.dis_d_out = self.discriminator.batch_classify(self.dis_x)

		self.dis_n_correct = tf.equal(tf.round(self.dis_d_out),self.dis_y)
		self.dis_accuracy = tf.reduce_mean(tf.cast(self.dis_n_correct, tf.float32))
		self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_d_out,labels=self.dis_y))


	## TRAINERS
	def train_generator_mle(self):
		"""
		Maximum Likelihood pretraining for generator
		"""
		print('\nStarting Generator MLE Training...')

		for epoch in range(self.mle_epochs):
			print('epoch {} :'.format(epoch+1),end='')
			sys.stdout.flush()

			total_loss = 0
			n_batches = 0

			for i in range(0,self.n_samples-self.batch_size,self.batch_size):
				n_batches += 1

				batches_data = self.data[i:i+self.batch_size]
				inp, target = self._prep_generator_batch_py(batches_data)

				feed_dict = {
				self.mle_x : inp,
				self.mle_y : target,
				}

				_, batch_loss = self.sess.run([self.mle_train,self.mle_loss],feed_dict)
				total_loss += batch_loss

				if (i / self.batch_size) % ceil(int(self.n_samples/self.batch_size) / 20) == 0:
					print('.',end='')
					sys.stdout.flush()

			## PRINT SAMPLES
			total_loss = total_loss / n_batches / self.seq_length
			print('Average MSE Loss per sample: {}'.format(total_loss))

		print('Done!')


	def pretrain_discriminator(self):
		print('\nStarting Discriminator Pre-Training...')
		self.train_discriminator(self.pt_steps,self.pt_epochs)
		print('Done!')


	def adversarial_training(self):
		for epoch in range(self.adv_epochs):
			print('\n--------\nEPOCH {}\n--------'.format(epoch+1))

			# Generator
			print('\nAdversarial Training Generator : ', end='')
			sys.stdout.flush()
			self.train_generator_PG(1)

			# Discriminator
			print('\nAdversarial Training Discriminator : ')
			self.train_discriminator(self.at_steps,self.at_epochs)


	def train_generator_PG(self, num_batches):
		"""
		The generator is trained using policy gradients, using the reward from the discriminator.
		Training is done for num_batches batches.
		"""
		for batch in range(num_batches):
			self.sess.run(self.pg_train)


	def train_discriminator(self,d_steps,epochs):
		# Train
		for d_step in range(d_steps):
			print('Retrieving Samples...')
			samples = self.sample(self.batch_size*int(self.n_samples/self.batch_size))
			dis_inp, dis_target = self._prep_discriminator_data(self.data, samples)

			for epoch in range(epochs):
				print('d-step {0} epoch {1} : '.format(d_step + 1, epoch + 1), end='')
				sys.stdout.flush()

				total_loss = 0
				total_acc = 0
				n_batches = 0

				for i in range(0,2*(self.n_samples-self.batch_size),self.batch_size):
					n_batches += 1

					inp, target = dis_inp[i:i + self.batch_size], dis_target[i:i + self.batch_size]

					feed_dict = {
					self.dis_x : inp,
					self.dis_y : target,
					}

					_, batch_loss, acc = self.sess.run([self.dis_train,self.dis_loss,self.dis_accuracy],feed_dict=feed_dict)

					total_loss += batch_loss
					total_acc += acc

					if (i / self.batch_size) % ceil(int(2*self.n_samples/self.batch_size) / 10) == 0:
						print('.', end='')
						sys.stdout.flush()

				total_loss /= n_batches
				total_acc /= n_batches

				print(' average_loss = {0}, train_acc = {1}'.format(total_loss, total_acc))


	def sample(self,num_samples):
		return self.sess.run(self.generator.sample(num_samples))


	## HELPERS
	def _prep_generator_batch_py(self,samples):
		inp = np.array(samples)
		target = np.array(samples)
		inp[:, 0] = self.start_return
		inp[:, 1:] = target[:, :-1]
		return inp, target

	def _prep_generator_batch_tf(self,samples):
		n, seq_length = self._shape(samples)

		target = samples

		init_returns = tf.constant(self.start_return,shape=[n],dtype=tf.float32)
		init_returns = tf.reshape(init_returns,shape=[n,1])

		inp = tf.concat([init_returns, target[:, :seq_length - 1]],1)
		return inp, target

	def _batchwise_sample(self,num_samples):
		samples = []
		for i in range(int(num_samples/self.batch_size)):
			sample = self.sample(self.batch_size) # : batch_size * seq_length
			samples += sample
		return samples # ~num_samples * seq_length

	def _prep_discriminator_data(self,pos_samples,neg_samples):
		neg_size = len(neg_samples)
		pos_samples = pos_samples[:neg_size]
		pos_size = len(pos_samples)

		pos_samples = np.array(pos_samples)
		neg_samples = np.array(neg_samples)

		inp = np.concatenate((pos_samples,neg_samples))
		target_pos = [1 for i in range(pos_size)]
		target_neg = [0 for i in range(neg_size)]
		target = np.array(target_pos+target_neg)

		shuffle_indices = np.random.permutation(np.arange(len(target)))
		inp = inp[shuffle_indices]
		target = target[shuffle_indices]

		return inp, target

	def _shape(self,tensor):
		size_0, size_1 = tensor.get_shape()
		arr = np.array([size_0,size_1], dtype=np.int32)
		return arr[0], arr[1]




