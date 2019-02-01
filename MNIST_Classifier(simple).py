import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Mnist_data", one_hot= True)

batch_size = 128
n_batch = mnist.train.num_examples//batch_size

x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])
keep_prob = tf.placeholder(tf.float32)
lrn_rate = tf.Variable(0.001, dtype=tf.float32)

#L1
W_L1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1  ))
B_L1 = tf.Variable(tf.zeros([1,500])+ 0.1)
h1 = tf.nn.relu(tf.matmul(x, W_L1) + B_L1)
h1_dropout = tf.nn.dropout(h1,keep_prob)

#L2
W_L2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
B_L2 = tf.Variable(tf.zeros([1,300])+ 0.1)
h2 = tf.nn.relu(tf.matmul(h1_dropout, W_L2) + B_L2)
h2_dropout =  tf.nn.dropout(h2,keep_prob)
#L3
W_L3 = tf.Variable(tf.truncated_normal([300, 100], stddev=0.1))
B_L3 = tf.Variable(tf.zeros([1,100])+ 0.1)
h3 = tf.nn.relu(tf.matmul(h2_dropout, W_L3) + B_L3)
h3_dropout = tf.nn.dropout(h3,keep_prob)
#L4
W_L4 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
B_L4 = tf.Variable(tf.zeros([1,10])+ 0.1)
output = tf.nn.softmax(tf.matmul(h3_dropout, W_L4) + B_L4)





#quadratic loss function
#loss = tf.reduce_mean(tf.square(y - output))
#likelihood loss function    tf.nn.softmax_cross_entropy_with_logits
#loss function for sigmoid   tf.nn.sigmoid_cross_entropy_with_logits
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))
train = tf.train.AdamOptimizer(lrn_rate).minimize(loss)
# train = tf.train.AdamOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.arg_max(output, 1), tf.arg_max(y, 1)) #return the position maximum value in the tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lrn_rate, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob:1})
        print("The accuracy of test data  " + str(epoch) + "  is" + str(test_acc))
        print("The accuracy of trained data  " + str(epoch) + "  is" + str(train_acc))
