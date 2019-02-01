import tensorflow as tf
import numpy as np
xixihaha = tf.constant("Hello World")
# x_data = np.linspace(-1, 1, 500)[:,np.newaxis]
# print(x_data.shape)
with tf.Session() as sess:
    print(sess.run(xixihaha))

# a_list =[]
# for x in range(5):
#     print(x)
#     a_list.append(x)
#
# a_list[2:4] = [8,8]
# print(a_list)
