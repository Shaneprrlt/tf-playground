import tensorflow as tf
import numpy as np
import pdb

# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

graph1 = tf.Graph()
with graph1.as_default():

    # construct input data and model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # construct cost calculation using cross entropy
    y_ = tf.placeholder(tf.float32, [None, 10])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # use gradient descent optimizer to train the model
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # create training session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # perform model training
    total_epochs = 1000
    for _ in range(total_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # evaluate the accuracy of the model
    # using the mnist test data
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    score = sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels })
    print('accuracy score: %s'%(score))

    sess.close()
