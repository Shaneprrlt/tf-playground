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

    # instance of the actual op
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # construct cost calculation using cross entropy
    # TODO: learn more about how cross entropy works (http://colah.github.io/posts/2015-09-Visual-Information/)
    y_ = tf.placeholder(tf.float32, [None, 10])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # TODO: understand the definition of logits (seems to be the model's guessed output
    # [but maybe in only this context])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # use gradient descent optimizer to train the model
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # create training session
    # TODO: learn differences in various sessions
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

    # TODO: Better Understand Syntax For Building Operations
    # and how the TF api stiches these calls together

    # TODO: Learn better design patterns, perhaps OO patterns for
    # writing TF code, while I'm impressed how little code there was
    # to build a fully functioning neural net, this seems a bit like spaghetti code
    # and more complex models will probably need better organization

    # TODO: Learn how model variables can be persisted and reloaded
    # (and possibly even hosted online? after all, the point of this library
    # is to have a production-ready ML toolkit, what does a prod environment look like)

    # TODO: Learn the different types of neural networks (convolutional, recurrent, etc)
    # and what use cases they are best applied for

    # Notes (key takeaways) after implementing:

    # 1. An ann at bare minimum has an input layer with an input node for each piece of data that
    # is used, in this example there are 784 input nodes to correspond to the 784 pixels in each
    # image.

    # 2. In an ann, the two 'variables' that get tweaked are the weights
    # and biases. These variables correspond to the variables that can be manipulated
    # in order to find a more accurate prediction line (and comes from statistics)

    # 3. After passing forward through the network, the output is compared to the
    # 'actual' data that is expected and comes from the test data.
    # a loss (or cost) calculation is performed (eg. Least Squares, Cross Entropy)
    # that shows how far off the networks output was from the real thing

    # 4. The goal in training is to use an optimizing function (eg. Gradient Descent)
    # to minimize the error that is calculated in the previous step
