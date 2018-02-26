# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:04:13 2017

@author: YimingZhao
"""

import random
from Configuration import Configuration_CNN
from Configuration import Configuration_FC
import math
import tensorflow as tf
from model import model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.python.platform import gfile

class CNNclassifier(model):
    def __init__(self,isTest=5):
        model.__init__(self,'CNN','CNN network')
        Conf=Configuration_CNN()
        Conf.Return()
        Conf.Return_CNN()
        self.input_layer_input_w,self.input_layer_output_w, self.input_layer_output_b,self.core_layer_input_w,self.core_layer_output_w,self.core_layer_output_b,self.action_layer_input_w,self.action_layer_output_w,self.action_layer_output_b,self.batch_size,self.epoch,self.learning_rate,self.checkpoint_path=Conf.Return()

        self.filter_depth,self.filter_height,self.filter_width,self.in_channels,self.out_channels,self.stride=Conf.Return_CNN()
        if isTest:
            self.epoch=1
        np.random.seed(1337)
        tf.set_random_seed(1337) 
        
    def weight_variable(self, shape, myname, train):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        initial = tf.random_uniform(shape, minval=-0.01, maxval = 0.01)
        return tf.Variable(initial, name=myname, trainable=train)
        
    
        
        
    def build_model(self):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        #input image
        self.img = tf.placeholder(tf.float32, shape=(None, self.channels,self.time,self.img_size_x,self.img_size_y))
        #change input and output layer size according to the dataset
        self.action_layer_output_w=self.n_classes
        self.action_layer_output_b=self.n_classes
        self.in_channels=self.channels
        #on_hot class number
        self.yy = tf.placeholder(tf.float32, shape=(None, self.n_classes))
        #CNN network
        Img=tf.transpose(self.img, perm=[0,2,3,4,1])
        initial = tf.random_uniform((self.filter_depth, self.filter_height, self.filter_width, self.in_channels, self.out_channels), minval=-0.01, maxval =0.01)
        Filter=tf.Variable(initial, name="CNN", trainable=True)
        b_conv=tf.Variable(tf.random_uniform((1,self.out_channels), minval=-0.01, maxval = 0.01),trainable=True)
        
        Con_Img=tf.nn.conv3d(Img, Filter, self.stride,"SAME", name=None)+b_conv
        Con_Img=tf.transpose(Con_Img,perm=[0,4,1,2,3])
        
        self.core_layer_input_w=int(self.out_channels*(math.ceil((self.time)/self.stride[1]))*(math.ceil((self.img_size_x)/self.stride[2]))*(math.ceil((self.img_size_y)/self.stride[3])))
        #core_network
        self.core_layer_w=self.weight_variable((self.core_layer_input_w,self.core_layer_output_w), "core_layer_w", True)
        self.core_layer_b=self.weight_variable((1,self.core_layer_output_b),"core_layer_b",True)
        #action_network        
        self.action_layer_w=self.weight_variable((self.action_layer_input_w,self.action_layer_output_w), "action_layer_w", True)
        self.action_layer_b=self.weight_variable((1,self.action_layer_output_b),"action_layer_b",True)
        CNN_input=tf.reshape(Con_Img,(-1,self.core_layer_input_w))
        Hidden_layer=tf.nn.relu(tf.matmul(CNN_input,self.core_layer_w)+self.core_layer_b)
        return Hidden_layer        
        
    def fit(self,train_data,train_label):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        n_samples, self.channels, self.time, self.img_size_x, self.img_size_y = train_data.shape[0:5]

        self.n_classes = self.action_layer_output_w
        Hidden_layer = self.build_model()
        self.y_conv = tf.nn.softmax(tf.matmul(Hidden_layer, self.action_layer_w) + self.action_layer_b)

        with tf.Session() as sess:
            cross_entropy = -tf.reduce_mean(
                tf.multiply(self.yy, tf.log(self.y_conv)) + tf.multiply((1.0 - self.yy), tf.log((1.0 - self.y_conv+0.0001))))

            trainer = tf.train.AdamOptimizer(self.learning_rate)
            Grads_and_Vars = trainer.compute_gradients(cross_entropy)
            train_step = trainer.apply_gradients(grads_and_vars=Grads_and_Vars)

            correct_prediction = tf.abs(self.y_conv - self.yy) < 0.5

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            sess.run(tf.global_variables_initializer())

            train_data = train_data.tolist()
            train_label = train_label.tolist()
            train_label_mat = train_label_mat.tolist()
            if len(train_data) % self.batch_size != 0:
                for i in range(self.batch_size-len(train_data) % self.batch_size):
                    index = random.randint(0, len(train_data) - 1)
                    train_data.append(train_data[index])
                    train_label.append(train_label[index])
                    train_label_mat.append(train_label_mat[index])
            assert len(train_data) % self.batch_size == 0
            assert len(train_label_mat) == len(train_data)
            Index = [i for i in range(len(train_data))]
            for i in range(self.epoch):
                Total_acc = 0.0
                random.shuffle(Index)
                for j in range(int(len(train_data) / self.batch_size)):
                    eve_train_data = []
                    eve_train_label_mat = []
                    for mm in Index[j * self.batch_size:(1 + j) * self.batch_size]:
                        eve_train_data.append(train_data[mm])
                        eve_train_label_mat.append(train_label_mat[mm])
                    train_step.run(feed_dict={self.img: eve_train_data, self.yy: eve_train_label_mat})
                    train_accuracy = (accuracy.eval(feed_dict={self.img: eve_train_data, self.yy: eve_train_label_mat}))
                    Total_acc = Total_acc + train_accuracy
                print("Epoch %d,training accuracy %g" % (i + 1, Total_acc / float(len(train_data) / self.batch_size)))
            saver = tf.train.Saver() 
            folder = os.path.dirname(self.checkpoint_path)
            if not gfile.Exists(folder):
                gfile.MakeDirs(folder)
            saver.save(sess, self.checkpoint_path)
       
    def inference(self,test_data):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint_path)
            y_pred = tf.round(self.y_conv)
            y = y_pred.eval(feed_dict={self.img: test_data})
                
        return y
    
    def clean(self):
        tf.reset_default_graph()
        
                