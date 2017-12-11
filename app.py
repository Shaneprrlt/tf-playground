import tensorflow as tf
import numpy as np
import pdb

# Graph1: Learning Basic TF Constructs

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

# Graph2: Learning Basic TF Constructs

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

# Graph3: Running a Simple Linear Regression

graph3 = tf.Graph()
with graph3.as_default():
    logs_path = 'logs/graph3'

    # linear regression model definition
    W = tf.Variable([0.3], dtype=tf.float32)
    b = tf.Variable([-0.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = tf.add(tf.multiply(W, x), b)

    with tf.Session() as session:
        # initialize variables
        init = tf.global_variables_initializer()
        session.run(init)

        # run linear model
        writer = tf.summary.FileWriter(logs_path, graph=session.graph)
        output = session.run(linear_model, feed_dict={ x: [1, 2, 3, 4] })
        print(output)

    session.close()

# Graph4: Running a Linear Regression and Calculate Loss (Cost Function)

graph4 = tf.Graph()
with graph4.as_default():
    logs_path = 'logs/graph4'

    # linear regression model definition
    W = tf.Variable([0.3], dtype=tf.float32)
    b = tf.Variable([-0.3], dtype=tf.float32)
    input_data = tf.placeholder(tf.float32)
    linear_model = W * input_data + b

    # loss function
    expected_output = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - expected_output)
    loss_calc = tf.reduce_sum(squared_deltas)

    with tf.Session() as session:
        # initialize variables
        init = tf.global_variables_initializer()
        session.run(init)

        # run linear model and loss function
        writer = tf.summary.FileWriter(logs_path, graph=session.graph)
        loss = session.run(loss_calc, feed_dict={ input_data: [1,2,3,4], expected_output: [0,-1,-2,-3] })
        print(loss)

    session.close()

# Graph5: Minimize the Loss (Cost) by using Gradient Decent
# to Correct Linear Regression Model

graph5 = tf.Graph()
with graph5.as_default():
    logs_path = 'logs/graph5'

    # linear regression model definition
    W = tf.Variable([0.3], dtype=tf.float32)
    b = tf.Variable([-0.3], dtype=tf.float32)
    input_data = tf.placeholder(tf.float32)
    linear_model = W * input_data + b

    # loss function
    expected_output = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - expected_output)
    loss_calc = tf.reduce_sum(squared_deltas)

    # optimizer (minimizes loss using gradient descent)
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss_calc)

    with tf.Session() as session:
        # initialize variables
        init = tf.global_variables_initializer()
        session.run(init)

        writer = tf.summary.FileWriter(logs_path, graph=session.graph)

        in_data = [1,2,3,4]
        out_data = [0,-1,-2,-3]
        total_epochs = 1000

        # train the model
        for i in range(total_epochs):
            session.run(train, { input_data: in_data, expected_output: out_data })

        # evaluate model accuracy
        eval_W, eval_b, eval_loss = session.run([W, b, loss_calc], { input_data: in_data, expected_output: out_data })
        print("W: %s, b: %s, loss: %s"%(eval_W, eval_b, eval_loss))

    session.close()

# Graph6: Using TF Estimator API To Run Linear Regression
# instead of writing it out by hand like in Graph 5

graph_6 = tf.Graph()
with graph_6.as_default():
    logs_path = 'logs/graph6'

    # linear regression estimator with a single feature column
    feature_columns = [tf.feature_column.numeric_column('x', shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    # training data
    train_input_data = np.array([1.0,2.0,3.0,4.0])
    train_expected_output_data = np.array([0.0,-1.0,-2.0,-3.0])

    # evaluation data
    eval_input_data = np.array([2.0,5.0,8.0,1.0])
    eval_expected_output_data = np.array([-1.01,-4.1,-7.0,0.0])

    # create input functions from numpy array data
    input_fn = tf.estimator.inputs.numpy_input_fn({ 'x': train_input_data }, train_expected_output_data, batch_size=4, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn({ 'x': train_input_data }, train_expected_output_data, batch_size=4, num_epochs=1000, shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn({ 'x': eval_input_data }, eval_expected_output_data, batch_size=4, num_epochs=1000, shuffle=False)

    # evalulate the accuracy of the model
    with tf.Session() as session:
        writer = tf.summary.FileWriter(logs_path, graph=session.graph)

        # train our linear regressor over 1000 epochs
        estimator.train(input_fn=input_fn, steps=1000)

        # evaluate the model
        train_metrics = estimator.evaluate(input_fn=train_input_fn)
        eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
        print('train metrics %r'%(train_metrics))
        print('eval metrics %r'%(eval_metrics))

    session.close()
