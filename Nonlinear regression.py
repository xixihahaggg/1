import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#create data point

#[:,np.newaxis]把数据点做成矩阵形式
x_data = np.linspace(-1, 1, 500)[:, np.newaxis]
noise = np.random.normal(0, 0.2, x_data.shape)
y_data = np.square(x_data) + noise



#define two placeholder to feed in
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#define hidden layer

#It has 10 hidden node
Weight_L1 = tf.Variable(tf.random_uniform([1, 10]))
Bias_L1 = tf.Variable(tf.zeros([1, 10]) + 0.01)
#It is a 1 by 10 matrix ,every row is a data// use Activation function
L1_output = tf.nn.tanh(tf.matmul(x, Weight_L1) + Bias_L1)

#define output layer
#let 1 by 11 timns 10 by 1 and produce the prediction
Weight_L2 = tf.Variable(tf.random_uniform([10, 1]))
Bias_L2 = tf.Variable(tf.zeros([1, 1]) + 0.01)
prediction = tf.nn.tanh(tf.matmul(L1_output, Weight_L2) + Bias_L2)

#loss function
loss = tf.reduce_mean(tf.square(y - prediction))
#gradien descent
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# train = tf.train.AdamOptimizer(0.001).minimize(loss)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    init = tf.global_variables_initializer()

    sess.run(init)
    for _ in range(5000):
        sess.run(train, feed_dict={x: x_data, y: y_data})

    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    print(sess.run(fetches=[Weight_L1, Bias_L1, Weight_L2, Bias_L2]))

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, "r", lw = 3)
    plt.grid()
    plt.show()