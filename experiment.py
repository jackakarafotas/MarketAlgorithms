import tensorflow as tf
import datetime as dt 
import math
from data_loader import DataLoader 
# import matplotlib.pyplot as plt 

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

BATCH_SIZE = 20
seq_length = 250
z_dim = 500
LEARNING_RATE = 1E-4

## Data
date_start = dt.datetime(1998,1,1)
date_end = dt.datetime(2017,12,31)

data = DataLoader(date_start,date_end)
data.scrape_tickers()
data.get_data(data.tickers)
data.batch_returns()

## GENERATOR
def generator(batch_size,z_dim,seq_length,reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()

	z = tf.truncated_normal([batch_size,z_dim],mean=0,stddev=1,name='z')

	# layer 1
	g_w1 = tf.get_variable('g_w1',[z_dim,2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
	g_b1 = tf.get_variable('g_b1',[2048],initializer=tf.truncated_normal_initializer(stddev=0.01))
	g1 = tf.matmul(z,g_w1) + g_b1
	g1 = tf.contrib.layers.batch_norm(g1,epsilon=1e-5,scope='bn1')
	g1 = tf.tanh(g1)

	# layer 2
	g_w2 = tf.get_variable('g_w2',[2048,1024],initializer=tf.truncated_normal_initializer(stddev=0.01))
	g_b2 = tf.get_variable('g_b2',[1024],initializer=tf.truncated_normal_initializer(stddev=0.01))
	g2 = tf.matmul(g1,g_w2) + g_b2
	g2 = tf.contrib.layers.batch_norm(g2,epsilon=1e-5,scope='bn2')
	g2 = tf.tanh(g2)
	g2 = tf.nn.dropout(g2, 0.8)

	# ouput layer
	g_w3 = tf.get_variable('g_w3',[1024,seq_length],initializer=tf.truncated_normal_initializer(stddev=0.01))
	g_b3 = tf.get_variable('g_b3',[seq_length],initializer=tf.truncated_normal_initializer(stddev=0.01))
	return tf.nn.tanh(tf.matmul(g2,g_w3) + g_b3)


## DISCRIMINATOR
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def avg_pool(x,pool_size):
	return tf.nn.avg_pool(x,ksize=[1,pool_size,1,1],strides=[1,pool_size,1,1],padding="SAME")

def discriminator(x,seq_length,reuse=False):
	# x shape = [batch, seq_length (height), 1 (width), 1 (channels)]
	if reuse:
		tf.get_variable_scope().reuse_variables()

	x_r = tf.reshape(x,shape=[-1,seq_length,1,1])

	# layer 1
	d_w1 = tf.get_variable('d_w1',[5,1,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d_b1 = tf.get_variable('d_b1',[32],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d_pool_size1 = 2
	d1 = avg_pool(tf.nn.relu(conv2d(x_r,d_w1)+ d_b1),d_pool_size1)

	# layer 2
	d_w2 = tf.get_variable('d_w2',[5,1,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d_b2 = tf.get_variable('d_b2',[64],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d_pool_size2 = 5
	d2 = avg_pool(tf.nn.relu(conv2d(d1,d_w2)+ d_b2),d_pool_size2)

	# layer 3
	in_dim = int(64 * (seq_length/(d_pool_size1*d_pool_size2)))
	d_w3 = tf.get_variable('d_w3',[in_dim,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d_b3 = tf.get_variable('d_b3',[1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d3 = tf.nn.relu(tf.matmul(tf.reshape(d2,[-1,in_dim]),d_w3) + d_b3)

	# output layer
	d_w4 = tf.get_variable('d_w4',[1024,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
	d_b4 = tf.get_variable('d_b4',[1],initializer=tf.truncated_normal_initializer(stddev=0.1))
	return tf.matmul(d3,d_w4) + d_b4

## Graph
CYCLES = len(data.batched_returns)

x = tf.placeholder(tf.float32,[None,seq_length],name='x')

with tf.variable_scope(tf.get_variable_scope()):
	Gz = generator(BATCH_SIZE,z_dim,seq_length)
	Dx = discriminator(x,seq_length)
	Dg = discriminator(Gz,seq_length,reuse=True)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.ones_like(Dg)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

with tf.variable_scope(tf.get_variable_scope()):
	d_trainer_real = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss_real,var_list=d_vars)
	d_trainer_fake = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss_fake,var_list=d_vars)
	g_trainer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss,var_list=g_vars)


## Train
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
gLoss = 0 
dLoss_real, dLoss_fake = 1, 1
d_real_count, d_fake_count, g_count = 0, 0, 0

for i in range(0,CYCLES,BATCH_SIZE):
	real_batch = data.batched_returns[i:i+BATCH_SIZE]

	## TRAIN
	if dLoss_fake > 0.6:
		# Train Discriminator on generated images
		_, dLoss_real, dLoss_fake, gLoss = sess.run([d_trainer_fake,d_loss_real,d_loss_fake,g_loss],{x:real_batch})
		d_fake_count += 1

	if gLoss > 0.5:
		# Train the generator
		_, dLoss_real, dLoss_fake, gLoss = sess.run([g_trainer,d_loss_real,d_loss_fake,g_loss],{x:real_batch})
		g_count += 1

	if dLoss_real > 0.45:
		# If the discriminator classifies real images as fake, train on real values
		_, dLoss_real, dLoss_fake, gLoss = sess.run([d_trainer_real,d_loss_real,d_loss_fake,g_loss],{x:real_batch})
		d_real_count += 1

	## Save / plot
	if i % 1000 == 0:
		with tf.variable_scope(tf.get_variable_scope()) as scope:
			graphs = sess.run(generator(3,z_dim,seq_length,reuse=True))
			d_result = sess.run(discriminator(x,seq_length),{x:graphs})

		print("\nTRAINING STEP:",i,"OUT OF",CYCLES,"AT",dt.datetime.now())
		for j in range(3):
			print("Discriminator classification:",sigmoid(d_result[j][0]))
			print(graphs[j][:5])

	if i % 5000 == 0:
		save_path = saver.save(sess,"models/pretrained_gan.ckpt",global_step = i)
		print("\nsaved to {0}".format(save_path))







	