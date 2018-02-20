import tensorflow as tf
import numpy as np


class Q_function():
    def __init__(self, batch_size=20, n_frame=4,action_num=6):
       
        self.n_frame=n_frame
        

        self.batch_size=batch_size
        self.action_num=action_num

        

    def weight_variable(self, shape, myname, train):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        initial = tf.random_uniform(shape, minval=-0.0001, maxval = 0.0001)
        return tf.Variable(initial, name=myname, trainable=train)


    def CNN_part(self):
        CNN_output=[]
        self.Input_tensor=tf.placeholder(tf.float32, shape=(None, self.n_frame, 7056))
        self.CNN_input=tf.transpose(self.Input_tensor,[0,2,1])
        self.CNN_input=tf.reshape(self.CNN_input,[-1,84,84,self.n_frame])
        self.conv1 = tf.layers.conv2d(inputs=self.CNN_input, filters=32, kernel_size=(8, 8), strides=(4, 4),padding="valid",activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, kernel_size=(4, 4), strides=(2, 2),padding="valid",activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, kernel_size=(3, 3), strides=(1, 1),padding="valid", activation=tf.nn.relu)
        CNN_output_shape=self.conv3.get_shape().as_list()
        CNN_output=tf.reshape(self.conv3,shape=(-1,CNN_output_shape[1]*CNN_output_shape[2]*CNN_output_shape[3]))
        self.FC_layer_w = self.weight_variable((CNN_output_shape[1]*CNN_output_shape[2]*CNN_output_shape[3], 512), "FC_layer_w", True)
        self.FC_layer_b = self.weight_variable((1,512),"FC_layer_b",True)
        self.final_layer_w=self.weight_variable((512,6),"Final_layer_w",True)
        self.final_layer_b=self.weight_variable((1,6),"Final_layer_b",True)
        FC_layer=tf.nn.relu(tf.matmul(CNN_output,self.FC_layer_w)+self.FC_layer_b)
        Final_layer=tf.matmul(FC_layer,self.final_layer_w)+self.final_layer_b
        return Final_layer
