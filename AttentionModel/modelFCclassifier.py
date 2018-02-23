# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:27:13 2017

@author: yimingzhao
"""
import random
from Configuration import Configuration_FC
import tensorflow as tf
from model import model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.python.platform import gfile
import gc


class FCclassifier(model):
    def __init__(self,isTest=5):
        model.__init__(self,'FC','Fully connective network')
        Conf=Configuration_FC()
        Conf.Return()
        self.input_layer_input_w,self.input_layer_output_w, self.input_layer_output_b,self.core_layer_input_w,self.core_layer_output_w,self.core_layer_output_b,self.action_layer_input_w,self.action_layer_output_w,self.action_layer_output_b,self.batch_size,self.epoch,self.learning_rate,self.checkpoint_path=Conf.Return()

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
        self.input_layer_input_w=self.channels*self.time*self.img_size_x*self.img_size_y 
        self.action_layer_output_w=self.n_classes  
        self.action_layer_output_b=self.n_classes
        #on_hot class number
        self.label=tf.placeholder(tf.int64, shape=(self.batch_size, 1))
        self.one_hot= tf.one_hot(indices=self.label, depth=self.n_classes)
        self.yy = tf.squeeze(self.one_hot, [1])


        #inout network        
        self.input_layer_w=self.weight_variable((self.input_layer_input_w,self.input_layer_output_w), "input_layer_w", True)
        self.input_layer_b=self.weight_variable((1,self.input_layer_output_b),"input_layer_b",True)
        #core_network
        self.core_layer_w=self.weight_variable((self.core_layer_input_w,self.core_layer_output_w), "core_layer_w", True)
        self.core_layer_b=self.weight_variable((1,self.core_layer_output_b),"core_layer_b",True)
        #action_network        
        self.action_layer_w=self.weight_variable((self.action_layer_input_w,self.action_layer_output_w), "action_layer_w", True)
        self.action_layer_b=self.weight_variable((1,self.action_layer_output_b),"action_layer_b",True)
        
        FC_input=tf.reshape(self.img,(-1,self.input_layer_input_w))
        Input_layer=tf.nn.relu(tf.matmul(FC_input,self.input_layer_w)+self.input_layer_b)
        
        Hidden_layer=tf.nn.relu(tf.matmul(Input_layer,self.core_layer_w)+self.core_layer_b)
        return Hidden_layer        
        
    def fit(self,train_data,train_label):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        n_samples, self.channels, self.time,self.img_size_x,self.img_size_y= train_data.shape[0:5]


        self.n_classes = self.action_layer_output_w
        Hidden_layer=self.build_model()
        self.y_conv =tf.nn.softmax(tf.matmul(Hidden_layer,self.action_layer_w)+self.action_layer_b)
        with tf.Session() as sess:


            cross_entropy = -tf.reduce_mean(tf.reduce_mean(tf.multiply(tf.log(self.y_conv+0.0001),self.yy),axis=1),axis=0)
            trainer=tf.train.AdamOptimizer(self.learning_rate)
            Grads_and_Vars = trainer.compute_gradients(cross_entropy)
            train_step=trainer.apply_gradients(grads_and_vars=Grads_and_Vars)
            correct_prediction = tf.abs(self.y_conv-self.yy)<0.5
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            sess.run(tf.global_variables_initializer())


            train_data = train_data.tolist()
            train_label = train_label.tolist()

            if len(train_data) % self.batch_size != 0:
                for i in range(self.batch_size-len(train_data) % self.batch_size):
                    index = random.randint(0, len(train_data) - 1)
                    train_data.append(train_data[index])
                    train_label.append(train_label[index])


            assert len(train_data) % self.batch_size == 0


            for i in range(self.epoch):
                Total_acc = 0.0

                for j in range(int(len(train_data) / self.batch_size-1)):
                    image=train_data[j*self.batch_size:(1+j)*self.batch_size]
                    label=np.reshape(train_label[j*self.batch_size:(1+j)*self.batch_size],(self.batch_size,1))
                    train_step.run(feed_dict={self.img: image, self.label: label})
                    train_accuracy = (accuracy.eval(feed_dict={self.img: image, self.label: label}))

                    Total_acc = Total_acc + train_accuracy
                    if j%100==0:
                        print (Total_acc/j)
                print("Epoch %d,training accuracy " % (i + 1))
                print  (Total_acc / float(len(train_data) / self.batch_size))
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
            print (y)
        return y
    
    def clean(self):
        tf.reset_default_graph()
        
                