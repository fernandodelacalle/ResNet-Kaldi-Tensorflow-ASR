import tensorflow as tf
import numpy as np

class RestNet(object):

    def __init__(self):
        print 'Init cnn layer'

    def __call__(self, inputs, is_training=False, reuse=False, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=reuse):        
            with tf.variable_scope('prep_data_l1', reuse=reuse):
                print inputs.get_shape()
                inputs_img = tf.reshape(inputs, tf.pack( [ tf.shape(inputs)[0] , 17, 64, 1] )  ) 
            
            
            x = self()

            x = self.convolution(inputs_img, 'first_conv_l1', 1, 64, 7, 7, reuse, is_training)
            with tf.variable_scope('pool_l1', reuse=reuse):
                x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


            x = self.residual('l1', x, 64, 64, [1, 1, 1, 1], reuse, is_training)

            x = self.residual('l2', x, 64, 128, [1, 1, 2, 1], reuse, is_training)

            x = self.residual('l3', x, 128, 128, [1, 1, 2, 1], reuse, is_training)

            x = self.residual('l4', x, 128, 256, [1, 2, 2, 1], reuse, is_training)

            x = self.residual('l5', x, 256, 256, [1, 2, 2, 1], reuse, is_training)

            #x = self.residual('l6', x, 256, 512, [1, 2, 2, 1], reuse, is_training)

            with tf.variable_scope('out_op', reuse=reuse):
                x = tf.nn.avg_pool(x, [1,3,2,1], [1,3,2,1], 'SAME')
                shape = x.get_shape().as_list()
                x = tf.reshape(x, tf.pack( [tf.shape(x)[0], shape[1]  * shape[2]  * shape[3]   ] ) )

            outputs = self.fully_connected('fcl', x, reuse, is_training)

        print 'Layer: ' + scope
        print 'Input: ' 
        print inputs.get_shape()
        print inputs_img.get_shape()
        print 'Outputs: ' 
        print outputs.get_shape()       
        return outputs


    def convolution(self, inputs_img, name_layer, in_dim, out_dim, t_conv_size, f_conv_size, reuse, is_training):
        with tf.variable_scope('parameters_'+name_layer, reuse=reuse):
            n = t_conv_size*f_conv_size*out_dim
            weights = tf.get_variable('weights_'+name_layer, [t_conv_size, f_conv_size, in_dim, out_dim],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            biases = tf.get_variable('biases_'+name_layer,   [out_dim],   initializer=tf.constant_initializer(0) )

        with tf.variable_scope('conv'+name_layer, reuse=reuse):
            conv = tf.nn.conv2d(inputs_img,  weights, [1, 1, 1, 1], padding='SAME')
            #print conv.get_shape()
            conv = tf.contrib.layers.batch_norm(conv,
                is_training=is_training,
                scope='batch_norm',
                reuse = reuse)
            hidden = tf.nn.relu(conv + biases)
        print 'hidden_'+ name_layer
        print hidden.get_shape() 
        return hidden


    def fully_connected(self, name_layer, x, reuse, is_training):

        print 'Layer: ' + name_layer
        print 'Input: ' 
        print  x.get_shape()

        output = 1000

        with tf.variable_scope(name_layer, reuse=reuse):
            with tf.variable_scope(name_layer +'_parameters', reuse=reuse):

                stddev = 1/(int(x.get_shape()[1])**0.5)

                weights = tf.get_variable(
                    'weights', [x.get_shape()[1], output],
                    initializer=tf.random_normal_initializer(stddev=stddev))

                biases = tf.get_variable(
                    'biases', [output],
                    initializer=tf.constant_initializer(0))

            x = tf.matmul(x, weights) + biases

            x = tf.contrib.layers.batch_norm(x,
                    is_training=is_training,
                    scope='batch_norm_'+name_layer,
                    reuse = reuse)
            
            x = tf.nn.relu(x)
            
        print 'Outputs: ' 
        print x.get_shape()
            
        return x


    def residual(self, name_layer, x, in_filter, out_filter, stride, reuse, is_training):
        """Residual unit with 2 sub layers."""
        
        print 'Layer: ' + name_layer
        print 'Input: ' 
        print  x.get_shape()

        orig_x = x
        with tf.variable_scope( name_layer + 'sub1', reuse=reuse):
            with tf.variable_scope('parameters_sub1_'+name_layer, reuse=reuse):
                n = 3*3*out_filter
                weights_sub1 = tf.get_variable('weights_sub1_'+name_layer, [3, 3, in_filter, out_filter],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
                biases_sub1 = tf.get_variable('biases_sub1'+name_layer,   [out_filter],   initializer=tf.constant_initializer(0.01) )

            with tf.variable_scope('conv_sub1'+name_layer, reuse=reuse):
                x = tf.nn.conv2d(x,  weights_sub1, stride, padding='SAME')
           
                x = tf.contrib.layers.batch_norm(x,
                    is_training=is_training,
                    scope='batch_norm_sub1_'+name_layer,
                    reuse = reuse)
                x = tf.nn.relu(x + biases_sub1)
            

        with tf.variable_scope( name_layer + 'sub2', reuse=reuse):
            with tf.variable_scope('parameters_sub2_'+name_layer, reuse=reuse):
                n = 3*3*out_filter
                weights_sub2 = tf.get_variable('weights_sub2_'+name_layer, [3, 3, out_filter, out_filter],  initializer = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
                biases_sub2 = tf.get_variable('biases_sub2_'+name_layer,   [out_filter],   initializer=tf.constant_initializer(0.01) )
            with tf.variable_scope('conv_sub2'+name_layer, reuse=reuse):
                x = tf.nn.conv2d(x,  weights_sub2, [1, 1, 1, 1], padding='SAME')
           
            with tf.variable_scope('sub_add'):
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'SAME')
                
                if in_filter != out_filter: 
                    if in_filter==1:
                        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],  [(out_filter - in_filter) // 2  ,  ((out_filter - in_filter) // 2) + 1 ]])
                    else:
                        orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2,
                              (out_filter - in_filter) // 2]])

                x += orig_x

                x = tf.contrib.layers.batch_norm(x,
                    is_training=is_training,
                    scope='batch_norm_sub1_'+name_layer,
                    reuse = reuse)
                x = tf.nn.relu(x + biases_sub2)
            
        print 'Outputs: ' 
        print x.get_shape()

        return x
