import tensorflow as tf



def random_choice(points, n_samples):
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(points)[1]), axis=0)
    indices = tf.random.categorical(uniform_log_prob, n_samples)
    indices = tf.squeeze(indices, 0, name="random_choice_ind")
    return tf.expand_dims(tf.gather(points[0], indices), axis=0)
