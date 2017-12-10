import tensorflow as tf
import numpy as np
import pdb

# build a computational graph
graph1 = tf.Graph()
with graph1.as_default():
    logs_path = 'logs/graph1'
    ## data
    const1 = tf.constant(3.0, dtype=tf.float32)
    const2 = tf.constant(4.0, dtype=tf.float32)
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    ## operations
    sum1 = tf.add(const1, const2)
    sum2 = tf.add(a,b)
    divide1 = tf.divide(sum2, sum1)
    # create a tensorflow session
    with tf.Session() as session:
        # run the computational graph
        writer = tf.summary.FileWriter(logs_path, graph=session.graph)
        output = session.run([divide1, sum2], { a: 16.0, b: 5.0 })
        print(output)
    # close the tf session
    session.close()

graph2 = tf.Graph()
with graph2.as_default():
    logs_path = 'logs/graph2'
    #data
    const1 = tf.constant(5.0, dtype=tf.float32)
    const2 = tf.constant(6.0, dtype=tf.float32)
    sum1 = tf.add(const1, const2)
    with tf.Session() as session:
        writer = tf.summary.FileWriter(logs_path, graph=session.graph)
        output = session.run([sum1])
        print(output)
    session.close()

graph3 = tf.Graph()
with graph3.as_default():
    logs_path = 'logs/graph3'

    # linear model definition
    W = tf.Variable([0.3], dtype=tf.float32)
    b = tf.Variable([-0.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = tf.add(tf.multiply(W, x), b)

    with tf.Session() as session:
        # initialize variables
        init = tf.global_variables_initializer()
        session.run(init)

        # run linear model
        output = session.run(linear_model, feed_dict={ x: [1, 2, 3, 4] })
        print(output)

    session.close()
