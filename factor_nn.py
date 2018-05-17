import tensorflow as tf 
import pandas as pd

def train_test_split(x_data,y_data,split_perc=0.7):
	x_split_value = int(len(x_data)*0.7)
	x_train = x_data[:x_split_value]
	x_test = x_data[x_split_value:]

	y_split_value = int(len(y_data)*0.7)
	y_train = y_data[:y_split_value]
	y_test = y_data[y_split_value:]

	return x_train, x_test, y_train, y_test

## CONSTANTS
BATCH_SIZE = 20
LR = 1e-4
N_EPOCH = 5

OUTPUT_DIM_1 = 16
OUTPUT_DIM_2 = 8

## Pre-Process Data
data = pd.read_csv('sp500_momentum.csv')
data = data[data.columns[1:]]
data.dropna(inplace=True)

def fix_data(val):
	if isinstance(val, float):
		return val
	else:
		return float(val.replace(',',''))

for col in data.columns:
	data[col] = data[col].apply(lambda val: fix_data(val))

y_data = data['Next 12 Month Return'].values.tolist()
x_data = data[data.columns[1:]].values.tolist()
INPUT_DIM = len(x_data[0])

x_batches = []
for i in range(0,len(x_data),BATCH_SIZE):
	new_batch = x_data[i:i+BATCH_SIZE]
	x_batches.append(new_batch)

y_batches = []
for i in range(0,len(y_data),BATCH_SIZE):
	new_batch = y_data[i:i+BATCH_SIZE]
	y_batches.append(new_batch)


x_train, x_test, y_train, y_test = train_test_split(x_data=x_batches,y_data=y_batches)

## Build Graph
# placeholders
x = tf.placeholder(tf.float32,[None,INPUT_DIM])
y = tf.placeholder(tf.float32,[None])
labels = tf.reshape(y,shape=[-1,1])

# Layers
W1 = tf.Variable(tf.random_normal([INPUT_DIM,OUTPUT_DIM_1], stddev=0.1))
b1 = tf.Variable(tf.random_normal([OUTPUT_DIM_1], stddev=0.1))
layer1 = tf.matmul(x,W1) + b1
layer1 = tf.nn.tanh(layer1)

# W2 = tf.Variable(tf.random_normal([OUTPUT_DIM_1,OUTPUT_DIM_2], stddev=0.1))
# b2 = tf.Variable(tf.random_normal([OUTPUT_DIM_2], stddev=0.1))
# layer2 = tf.matmul(layer1,W2) + b2
# layer2 = tf.nn.relu(layer2)

layer2 = tf.layers.dense(layer1,OUTPUT_DIM_2,activation=tf.nn.tanh)

W3 = tf.Variable(tf.random_normal([OUTPUT_DIM_2,1], stddev=0.1))
b3 = tf.Variable(tf.random_normal([1], stddev=0.1))
logits = tf.matmul(layer2,W3) + b3
logits = tf.nn.dropout(logits, 0.8)

# Loss and Train
loss = tf.losses.mean_squared_error(labels,logits)
train = tf.train.AdamOptimizer(LR).minimize(loss)

NUM_BATCHES_TRAIN = len(x_train)
NUM_BATCHES_TEST = len(x_test)

## TRAIN
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# training
	for epoch in range(N_EPOCH):
		epoch_loss = 0

		for i in range(NUM_BATCHES_TRAIN):
			x_batch = x_train[i]
			y_batch = y_train[i]

			feed_dict = {
			x : x_batch,
			y : y_batch
			}

			_, batch_loss = sess.run([train,loss],feed_dict = feed_dict)
			epoch_loss += batch_loss

		epoch_loss = epoch_loss / NUM_BATCHES_TRAIN
		print("Epoch:",epoch+1,"out of",N_EPOCH,"- Loss:",epoch_loss)

	# testing
	mse = 0
	for i in range(NUM_BATCHES_TEST):
		feed_dict = {x : x_test[i],y : y_test[i]}
		mse += loss.eval(feed_dict = feed_dict)

		if i%(int(NUM_BATCHES_TEST/5)) == 0:
			pred = logits.eval(feed_dict = feed_dict)
			print(x_test[i])
			print()
			print(pred)
			print()
			print(y_test[i])
			print()

	mse = mse / NUM_BATCHES_TEST
	print("Test MSE:",mse)









