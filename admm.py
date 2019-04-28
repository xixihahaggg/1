import tensorflow as tf
import numpy as np
import martrixapproximation
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Mnist_data", one_hot=True)

batch_size = 128
n_batch = mnist.train.num_examples // batch_size

rho = 0.1

# define two placeholder to feed in
x = tf.placeholder(tf.float32, [None, 784], name='input')
y = tf.placeholder(tf.float32, [None, 10], name='label')
# define hidden layer

# It has 10 hidden node
Weight_L1 = tf.Variable(tf.random_normal([784, 200]))
F_1 = tf.Variable(Weight_L1.initialized_value(), name='F1', trainable=False)
Lambda_1 = tf.Variable(tf.zeros([784, 200]), trainable=False)
Bias_L1 = tf.Variable(tf.zeros([1, 200]) + 0.01)
# It is a 1 by 10 matrix ,every row is a data// use Activation function
L1_output = tf.nn.tanh(tf.matmul(x, Weight_L1) + Bias_L1)
L1_sparse = tf.nn.tanh(tf.matmul(x, F_1) + Bias_L1)
# define output layer
# let 1 by 11 timns 10 by 1 and produce the prediction
Weight_L2 = tf.Variable(tf.random_normal([200, 50]))
F_2 = tf.Variable(Weight_L2.initialized_value(), name='F2', trainable=False)
Lambda_2 = tf.Variable(tf.zeros([200, 50]), trainable=False)
Bias_L2 = tf.Variable(tf.zeros([1, 50]) + 0.01)
L2_output = tf.nn.tanh(tf.matmul(L1_output, Weight_L2) + Bias_L2)
L2_sparse = tf.nn.tanh(tf.matmul(L1_sparse, F_2) + Bias_L2)

Weight_L3 = tf.Variable(tf.random_normal([50, 10]))
F_3 = tf.Variable(Weight_L3.initialized_value(), name='F3', trainable=False)
Lambda_3 = tf.Variable(tf.zeros([50, 10]), trainable=False)
Bias_L3 = tf.Variable(tf.zeros([1, 10]) + 0.01)
prediction = tf.nn.softmax(tf.matmul(L2_output, Weight_L3) + Bias_L3)
pred = tf.nn.tanh(tf.matmul(L2_sparse, F_3) + Bias_L3)


def sparse_estimate(V):
    zero_ind = tf.equal(V, 0)
    P = tf.reduce_mean(tf.cast(zero_ind, tf.float32))
    return P


def f_norm(V):
    square_tensor = tf.square(V)
    frobenius_norm = tf.reduce_sum(square_tensor)
    return frobenius_norm


def trunc(W, F, Lambda, ratio):
    v1 = (1 / rho) * Lambda
    V = tf.add(W, v1)
    list = tf.layers.flatten(V)
    list1 = tf.reshape(list, [-1])
    list2 = tf.contrib.framework.sort(tf.abs(list1))
    length = tf.cast(tf.size(list2), tf.float32)
    index = tf.cast(length * ratio - 1, tf.int32)
    threshold = list2[index]
    # zero_ind = tf.greater_equal(tf.abs(V1), b)
    zero_ind = tf.greater_equal(tf.abs(V), threshold)
    # print(sess.run(zero_ind))
    mask = tf.cast(zero_ind, tf.float32)
    v = tf.multiply(V, mask)

    return tf.assign(F, v)


def toeplitz(W, F, Lambda):
    v1 = (1 / rho) * Lambda
    V = tf.add(W, v1)
    H = martrixapproximation.project2toeplitz(V)
    return tf.assign(F, H)


def update_dual(dual_variable, V1, V2):
    update = dual_variable + rho * (tf.subtract(V1, V2))
    return tf.assign(dual_variable, update)


# loss function
U1 = tf.subtract(F_1, 1 / rho * Lambda_1)
U2 = tf.subtract(F_2, 1 / rho * Lambda_2)
U3 = tf.subtract(F_3, 1 / rho * Lambda_3)
penalize_term = (rho / 2) * (f_norm(tf.subtract(Weight_L1, U1)) +
                             f_norm(tf.subtract(Weight_L2, U2)) + f_norm(tf.subtract(Weight_L3, U3)))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction) + penalize_term
# gradien descent
train = tf.train.AdamOptimizer(0.01).minimize(loss)
correct_prediction = tf.equal(tf.arg_max(prediction, 1),
                              tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

correct_pred = tf.equal(tf.arg_max(pred, 1),
                        tf.arg_max(y, 1))
acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # print(sess.run([Weight_L3, F_3, Lambda_3]))
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
            # sess.run([trunc(Weight_L1, F_1, Lambda_1, 0.5), trunc(Weight_L2, F_2, Lambda_2, 0.5),
            #           trunc(Weight_L3, F_3, Lambda_3, 0.5)])
            sess.run([toeplitz(Weight_L1, F_1, Lambda_1), toeplitz(Weight_L2, F_2, Lambda_2),
                      toeplitz(Weight_L3, F_3, Lambda_3)])
            sess.run([update_dual(Lambda_1, Weight_L1, F_1),
                      update_dual(Lambda_2, Weight_L2, F_2),
                      update_dual(Lambda_3, Weight_L3, F_3)])
            # print(sess.run([Weight_L3,F_3, Lambda_3]))
            # print(sess.run([Weight_L3, F_3, Lambda_3]))
            # summary, _ = sess.run([merged, train], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        # writer.add_summary(summary, epoch)
        # print(sess.run([Weight_L3, F_3, Lambda_3]))
        # print(sess.run(f_norm(tf.subtract(Weight_L3, F_3))))
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("The accuracy of test data  " + str(epoch) + "  is" + str(test_acc))
        print("The accuracy of trained data  " + str(epoch) + "  is" + str(train_acc))
        test_acc1 = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc1 = sess.run(acc, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Sparse accuracy of test data  " + str(epoch) + "  is" + str(test_acc1))
        print("Sparse accuracy of trained data  " + str(epoch) + "  is" + str(train_acc1))
        print(sess.run(F_3))

        # print('The saprsity of F_1 is {}'.format(sess.run(sparse_estimate(F_1))))
        # print('The saprsity of F_2 is {}'.format(sess.run(sparse_estimate(F_2))))
        # print('The saprsity of F_3 is {}'.format(sess.run(sparse_estimate(F_3))))
