import tensorflow as tf
def create_placeholders(n_H0,n_W0,n_C0,n_y):
	X=tf.placeholder(tf.float32,[None,28*28])
	x_image = tf.reshape(X, [-1, 28, 28, 1])
	Y=tf.placeholder(tf.float32,[None,n_y])
	return x_image,Y
	