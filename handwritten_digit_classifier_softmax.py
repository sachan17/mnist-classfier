#helplink https://www.tensorflow.org/tutorials/mnist/beginners/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc
# done using onlu=y softmax 

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

'''
load mnist data 55000 train, 10000 test, 5000 validation 
mnist.train.images [55000 784] each images of 28 * 28
mnist.train.labels [55000 10] 1 for the actual class, 0 for rest
same with test mnist.test.images
'''
mnist = input_data.read_data_sets("MNIST-data/", one_hot = True)


import tensorflow as tf

# placeholder of input matrix with each row having 784 pixel values
# None represent it can have any number of rows
#y_ contains the actual labels 
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
#W for weights for each pixel for each class
#b for bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#y is will predicted values for each class
y = tf.nn.softmax(tf.matmul(x, W) + b)

#applying cross entropy as cost function
# Cost(y) = - Ei (yi'log(yi))
#numerically this is unstable
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
#more stable version
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#training
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x : batch_xs, y_ : batch_ys})

#accuracy test on test data
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print(sess.run(accuracy, feed_dict={x : mnist.test.images, y_:mnist.test.labels}))


# show off
def display(i):
	'''
	displays test data image of 28*28
	and its actual label list
	'''
	img = mnist.test.images[i]
	plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
	plt.show()

from random import randint
for each in range(10):
	i = randint(0, 10000)
	ans = tf.argmax(y, 1)
	print('Predicted value', sess.run(ans, feed_dict={x : [mnist.test.images[i]]}))
	display(i)