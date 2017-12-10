import tensorflow as tf

def leaky_relu(input_x, n_slop=0.2):
    return tf.maximum(input_x*n_slop, input_x)

