import tensorflow as tf
import numpy as np

class CnnVd6(object):
    def __init__(self):
        print 'Init cnn layer'


    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):        
            with tf.variable_scope('prep_data_l1', reuse=reuse):
                print inputs.get_shape()
                # For ddeltas features, the input map is examples x time x freq x 3
		#inputs_img = tf.reshape(inputs, tf.pack( [ tf.shape(inputs)[0] , 11, 3, 40] )  ) 
                #inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )  
		#inputs_img = inputs_img[:,:,:,0]
                #inputs_img = tf.reshape(inputs_img, tf.pack( [ tf.shape(inputs_img)[0] , 11, 40, 1] )  ) 
		        # For nodelta features
                inputs_img = tf.reshape(inputs, tf.pack( [ tf.shape(inputs)[0] , 11, 40, 1] )  ) 
                print inputs_img.get_shape()

            hidden = self.convolution(inputs_img, 'conv_l1', 1, 64, 1, 3, reuse, is_training)
            hidden = self.convolution(hidden, 'conv_l2', 64, 64, 3, 3, reuse, is_training)
            with tf.variable_scope('pool_l2', reuse=reuse):
                pool = tf.nn.max_pool(hidden, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

            hidden = self.convolution(pool, 'conv_l3', 64, 128, 3, 3, reuse, is_training)
            hidden = self.convolution(hidden, 'conv_l4', 128, 128, 3, 3, reuse, is_training)
            with tf.variable_scope('pool_l4', reuse=reuse):
                pool = tf.nn.max_pool(hidden, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')

            hidden = self.convolution(pool, 'conv_l5', 128, 256, 3, 3, reuse, is_training)
            hidden = self.convolution(hidden, 'conv_l6', 256, 256, 3, 3, reuse, is_training)

            with tf.variable_scope('out_op', reuse=reuse):
                shape = hidden.get_shape().as_list()
                outputs = tf.reshape(hidden, tf.pack( [tf.shape(hidden)[0], shape[1]  * shape[2]  * shape[3]   ] ) )

        print 'Layer: ' + scope
        print 'Input: ' 
        print  inputs.get_shape()
        print 'Outputs: ' 
        print outputs.get_shape()
        return outputs


    def convolution(self, inputs_img, name_layer, in_dim, out_dim, t_conv_size, f_conv_size, reuse, is_training):
        with tf.variable_scope('parameters_'+name_layer, reuse=reuse):
            n = t_conv_size*f_conv_size*out_dim
            weights = tf.get_variable('weights_'+name_layer, [t_conv_size, f_conv_size, in_dim, out_dim],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            biases = tf.get_variable('biases_'+name_layer,   [out_dim],   initializer=tf.constant_initializer(0) )

        with tf.variable_scope('conv'+name_layer, reuse=reuse):
            conv = tf.nn.conv2d(inputs_img,  weights, [1, 1, 1, 1], padding='VALID')
            #print conv.get_shape()
            conv = tf.contrib.layers.batch_norm(conv,
                is_training=is_training,
                scope='batch_norm',
                reuse = reuse)
            hidden = tf.nn.relu(conv + biases)
        print 'hidden_'+ name_layer
        print hidden.get_shape() 
        return hidden
