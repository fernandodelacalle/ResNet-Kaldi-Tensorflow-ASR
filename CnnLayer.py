import tensorflow as tf
import numpy as np

class CnnLayer(object):

    def __init__(self):
        print 'Init cnn layer'

   
    def __call__(self, inputs, is_training=False, reuse=False, scope=None):        
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
            
            print 'Layer: ' + scope
            print 'Input: ' 
            print inputs.get_shape()

            with tf.variable_scope('prep_data_l1', reuse=reuse):             
                inputs_img = tf.reshape(inputs, tf.pack( [ tf.shape(inputs)[0] , 11, 3, 40] )  ) 
                inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )  
    
            print 'Input Img: ' 
            print inputs_img.get_shape()

            hidden = self.convolution(inputs_img, 'conv_l1', 3, 256, 9, 9, reuse, is_training)

            with tf.variable_scope('pool_l1', reuse=reuse):
                pool = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 3, 1], padding='VALID')

            print 'poll_l1: ' 
            print pool.get_shape()

            hidden = self.convolution(pool, 'conv_l2', 256, 256, 3, 4, reuse, is_training)
            
            with tf.variable_scope('out_op', reuse=reuse):
                shape = hidden.get_shape().as_list()
                outputs = tf.reshape(hidden, tf.pack( [tf.shape(hidden)[0], shape[1]  * shape[2]  * shape[3]   ] ) )
         
            print 'Outputs: ' 
            print outputs.get_shape()
        
        return outputs

    def convolution(self, inputs_img, name_layer, in_dim, out_dim, t_conv_size, f_conv_size, reuse, is_training):
        with tf.variable_scope('parameters_'+name_layer, reuse=reuse):
            n = t_conv_size*f_conv_size*out_dim
            weights = tf.get_variable('weights_'+name_layer, [t_conv_size, f_conv_size, in_dim, out_dim],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            biases = tf.get_variable('biases_'+name_layer,   [out_dim],   initializer=tf.constant_initializer(0) )

        with tf.variable_scope('conv_'+name_layer, reuse=reuse):
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




    # def __call__(self, inputs, is_training=False, reuse=False, scope=None):
    #     '''
    #     Do the forward computation
    #     Args:
    #         inputs: the input to the layer
    #         is_training: whether or not the network is in training mode
    #         reuse: wheter or not the variables in the network should be reused
    #         scope: the variable scope of the layer
    #     Returns:
    #         The output of the layer
    #     '''

    #     with tf.variable_scope(scope or type(self).__name__, reuse=reuse):
    #         with tf.variable_scope('parameters', reuse=reuse):
    #                 f = 9
    #                 d_1 = 3
    #                 k = 256
    #                 n = f*f*k
    #                 weights_l1= tf.get_variable('weights_fc_1', [f, f, d_1, k],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #                 biases_l1 = tf.get_variable('biases_fc_1',   [k],   initializer=tf.constant_initializer(0) )
    #                 k = 256
    #                 n = f*f*k
    #                 weights_l2 = tf.get_variable('weights_fc_2', [3, 4, k, k],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #                 biases_l2 = tf.get_variable('biases_fc_2',   [k],   initializer=tf.constant_initializer(0) )





    #         print inputs.get_shape()
    #         inputs_img = tf.reshape(inputs, tf.pack( [ tf.shape(inputs)[0] , 11, 3, 40] )  ) 
    #         inputs_img = tf.transpose(inputs_img, [ 0 , 1, 3, 2 ] )  
    #         print inputs_img.get_shape()

    #         with tf.variable_scope('conv_l1', reuse=reuse):
    #             conv = tf.nn.conv2d(inputs_img,  weights_l1, [1, 1, 1, 1], padding='VALID')
    #             #print conv.get_shape()

    #             conv = tf.contrib.layers.batch_norm(conv,
    #                 is_training=is_training,
    #                 scope='batch_norm',
    #                 reuse = reuse)

    #             hidden = tf.nn.relu(conv + biases_l1)
    #             print hidden.get_shape()        

    #             pool = tf.nn.max_pool(hidden, ksize=[1, 1, 1, 1], strides=[1, 1, 3, 1], padding='VALID')
    #             print pool.get_shape()

    #         with tf.variable_scope('conv_l2', reuse=reuse):
    #             conv = tf.nn.conv2d(pool, weights_l2, [1, 1, 1, 1], padding='VALID')
    #             #print conv.get_shape()
            
    #             conv = tf.contrib.layers.batch_norm(conv,
    #                 is_training=is_training,
    #                 scope='batch_norm',
    #                 reuse = reuse)

    #             hidden = tf.nn.relu(conv + biases_l2)
    #             print hidden.get_shape()

    #             shape = hidden.get_shape().as_list()
    #             outputs = tf.reshape(hidden, tf.pack( [tf.shape(pool)[0], shape[1]  * shape[2]  * shape[3]   ] ) )
    #             print outputs.get_shape()

    #     print 'Layer: ' + scope
    #     print 'Input: ' 
    #     print  inputs.get_shape()
    #     print 'Outputs: ' 
    #     print outputs.get_shape()


    #     return outputs
