# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:50:34 2017

@author: YimingZhao
"""

import random
import tensorflow as tf
from model import model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.python.platform import gfile
from Configuration import Configuration_RandGlimpse
import math
from random import randint

class RandGlimpseclassifier(model):
    def __init__(self,isTest=5):
        model.__init__(self,'RandGlimpse','RandGlimpse network')
        Conf=Configuration_RandGlimpse()
        Conf.Return()
        self.input_layer_input_w,self.input_layer_output_w, self.input_layer_output_b,self.core_layer_input_w,self.core_layer_output_w,self.core_layer_output_b,self.action_layer_input_w,self.action_layer_output_w,self.action_layer_output_b,self.batch_size,self.epoch,self.learning_rate,self.checkpoint_path=Conf.Return()
        self.repeat,self.transfer_layer_input_w=Conf.Return_RNN()
        self.sensorX,self.sensorY,self.depth,self.hg_size,self.hl_size,self.glimpse_size=Conf.Return_RandGlimpse()
        if isTest:
            self.epoch=1
        np.random.seed(1337)
        tf.set_random_seed(1337) 
        
    def weight_variable(self, shape, myname, train):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        initial = tf.random_uniform(shape, minval=-0.01, maxval = 0.01)
        return tf.Variable(initial, name=myname, trainable=train)
        
        
    
    def glimpseSensor(self, img, normLoc):
        img=tf.reshape(img,(self.batch_size,self.channels,self.img_size_x,self.img_size_y))
        img=tf.transpose(img,perm=[0,2,3,1])
        loc = tf.round(((normLoc + 1) / 2.0) * [self.img_size_x,self.img_size_y])  # normLoc coordinates are between -1 and 1
        loc = tf.cast(loc, tf.int32)
        # process each image individually
        #print ("Glimpse one time")
        zooms = []
        for k in range(self.batch_size):
            imgZooms = []
            one_img = img[k,:,:,:]
            max_radius_x = int(math.ceil(self.sensorX/2) * (2 ** (self.depth - 1)))
            offset_x = int(2 * max_radius_x)
            max_radius_y = int(math.ceil(self.sensorY/2) * (2 ** (self.depth - 1)))
            offset_y = int(2 * max_radius_y)
            
            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset_x, offset_y, max_radius_x * 4 + self.img_size_x, max_radius_y * 4 + self.img_size_y)
            for j in range(self.channels):
                zoom_lin=[]
                for i in range(self.depth):
                    r_x = int(math.ceil(self.sensorX/2) * (2 ** (i)))
                #glimeps informati
                    d_raw_x = 2 * r_x
                    r_y = int(math.ceil(self.sensorY/2) * (2 ** (i)))
                #glimeps informati
                    d_raw_y = 2 * r_y
                    
                    d = tf.constant([d_raw_x,d_raw_y], shape=[2])
                    loc_k = loc[k,:]
                    
                    adjusted_loc = [offset_x,offset_y] + loc_k - [r_x,r_y] 
                    one_img2 = tf.reshape(one_img[:,:,j], (one_img.get_shape()[0].value,one_img.get_shape()[1].value))
                # crop image to (d x d)
                    zoom = tf.slice(one_img2, adjusted_loc, d)
                # resize cropped image to (sensorBandwidth x sensorBandwidth)
                    zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw_x, d_raw_y, 1)),(self.sensorX, self.sensorY))
                    zoom = tf.reshape(zoom, (self.sensorX, self.sensorY))
                    zoom_lin.append(zoom)
                imgZooms.append(zoom_lin)
            zooms.append(tf.stack(imgZooms))
        zooms = tf.stack(zooms)
        glimpse_input = tf.reshape(tf.squeeze(zooms), (self.batch_size,(self.sensorX*self.sensorY)*self.depth*self.channels ))
        normLoc=tf.reshape(normLoc,(self.batch_size,2))
        
        # the hidden units that process location & the input
        act_glimpse_hidden = tf.nn.relu(tf.matmul(glimpse_input, self.glimpse_layer_w) + self.glimpse_layer_b)
        act_loc_hidden = tf.nn.relu(tf.matmul(normLoc, self.loc_layer_w) + self.loc_layer_b)
        # the hidden units that integrates the location & the glimpses
        
        glimpseFeature1 = tf.nn.relu(tf.matmul(act_glimpse_hidden, self.Gli_glimpse_layer_w) + tf.matmul                                                                 (act_loc_hidden, self.Gli_loc_layer_w) + self.Gli_b)
        return glimpseFeature1
    
    
    
    
    def build_model(self):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        # set the learning rate
        lrDecayFreq = 200
        lrDecayRate = 0.99
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(self.learning_rate, self.global_step, lrDecayFreq, lrDecayRate,
                                             staircase=True)
       #input image
        self.img = tf.placeholder(tf.float32, shape=(None, self.channels,self.time,self.img_size_x,self.img_size_y))
        #change input and output layer size according to the dataset        
        Glimpse_input=self.channels*self.img_size_x*self.img_size_y 
        self.action_layer_output_w=self.n_classes  
        self.action_layer_output_b=self.n_classes
        #on_hot class number
        self.yy = tf.placeholder(tf.float32, shape=(None, self.n_classes))
        #Glimpse part
        self.loc_layer_w = self.weight_variable((2, self.hl_size), "loc_layer_w", True)
        self.loc_layer_b = self.weight_variable((1,self.hl_size), "loc_layer_b", True) 
        self.glimpse_layer_w = self.weight_variable(((self.depth * self.channels * (self.sensorX*self.sensorY)), self.hg_size), "glimpse_layer_w", True)
        self.glimpse_layer_b = self.weight_variable((1,self.hg_size), "glimpse_layer_b", True)
        self.Gli_glimpse_layer_w = self.weight_variable((self.hg_size, self.glimpse_size),"Gli_glimpse_layer_w",  True)
        self.Gli_loc_layer_w = self.weight_variable((self.hl_size, self.glimpse_size), "Gli_loc_layer_w", True)
        self.Gli_b = self.weight_variable((1,self.glimpse_size), "Num7", True)
        
        #transfer_layer network
        self.transfer_layer_w=self.weight_variable((self.transfer_layer_input_w,self.transfer_layer_input_w), "transfer_layer_w", True)
        #core_network
        self.core_layer_w=self.weight_variable((self.core_layer_input_w,self.core_layer_output_w), "core_layer_w", True)
        self.core_layer_b=self.weight_variable((1,self.core_layer_output_b),"core_layer_b",True)
        
        #action_network        
        self.action_layer_w=self.weight_variable((self.action_layer_input_w,self.action_layer_output_w), "action_layer_w", True)
        self.action_layer_b=self.weight_variable((1,self.action_layer_output_b),"action_layer_b",True)
        
        Hidden_layer=[]
        Hidden_layer.append(tf.random_uniform((self.batch_size,self.core_layer_output_w),minval=-0.01, maxval = 0.01))
        Locs=[]
        Locs.append(tf.random_uniform((self.batch_size, 2), minval=-0.2, maxval=0.2))
        count=0
        for i in range(self.time):
            Img=self.img[:,:,i,:,:]
            for j in range(self.repeat):
                Gli_input=tf.reshape(Img,(-1,Glimpse_input))
                Gli_input=self.glimpseSensor(Gli_input, Locs[count])
                Hidden_layer.append(tf.nn.relu(tf.matmul(Gli_input,self.core_layer_w)+self.core_layer_b+tf.matmul(Hidden_layer[count],self.transfer_layer_w)))
                Locs.append(tf.random_uniform((self.batch_size, 2), minval=-1.0, maxval=1.0))
                count=count+1
        return Hidden_layer[-1]        
        
    def fit(self,train_data,train_label):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        n_samples, self.channels, self.time,self.img_size_x,self.img_size_y= train_data.shape[0:5]
        train_label_mat = []
        for i in train_label:
            train_label_mat.append([i])
        train_label_mat = np.array(train_label_mat)

        self.n_classes = 1
        Hidden_layer = self.build_model()
        self.y_conv = tf.divide(1.0, tf.minimum(
            tf.add(1.0, tf.exp(-(tf.matmul(Hidden_layer, self.action_layer_w) + self.action_layer_b))), 1000.0))
        with tf.Session() as sess:
            cross_entropy = -tf.reduce_mean(
                tf.multiply(self.yy, tf.log(self.y_conv+0.0001)) + tf.multiply((1.0 - self.yy), tf.log((1.0 - self.y_conv+0.0001))))

            trainer = tf.train.AdamOptimizer(self.lr)
            train_step = trainer.minimize(cross_entropy , self.global_step)

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
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.checkpoint_path)
            test_data=test_data.tolist()
            Gener=self.batch_size-len(test_data)%self.batch_size
            Len=len(test_data)-1
            for i in range(Gener):
                test_data.append(test_data[randint(0,Len)])
            assert len(test_data)%self.batch_size==0
            y_pred = tf.round(self.y_conv)
            for i in range(int(len(test_data)/self.batch_size)):
                X=[]
                for j in range(self.batch_size):
                    X.append(test_data[i*self.batch_size + j])
                y = y_pred.eval(feed_dict={self.img: X})
                if i == 0:
                    print (1)
                    Y = y
                else:
                    Y = np.concatenate((Y,y), axis=0)
        return Y[:(Len+1)]
    
    def clean(self):
        tf.reset_default_graph()
        
                