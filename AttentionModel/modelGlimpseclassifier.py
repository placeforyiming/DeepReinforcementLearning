# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:16:48 2017

@author: YimingZhao
"""
import gc
import json
import random
import tensorflow as tf
from model import model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.python.platform import gfile
from Configuration import Configuration_Glimpse
import math
from random import randint
import csv
class Glimpseclassifier(model):
    def __init__(self,isTest=0):
        model.__init__(self,'Glimpse','Glimpse network')
        Conf=Configuration_Glimpse()
        Conf.Return()
        self.input_layer_input_w,self.input_layer_output_w, self.input_layer_output_b,self.core_layer_input_w,self.core_layer_output_w,self.core_layer_output_b,self.action_layer_input_w,self.action_layer_output_w,self.action_layer_output_b,self.batch_size,self.epoch,self.learning_rate,self.checkpoint_path=Conf.Return()
        self.repeat,self.transfer_layer_input_w=Conf.Return_RNN()
        self.sensorX,self.sensorY,self.depth,self.hg_size,self.hl_size,self.glimpse_size=Conf.Return_RandGlimpse()
        self.stdd,self.reward_ratio=Conf.Return_Glimpse()
        if isTest:
            self.epoch=1
        np.random.seed(1337)
        tf.set_random_seed(1337) 
        
    def weight_variable(self, shape, myname, train):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        initial = tf.random_uniform(shape, minval=-0.01, maxval = 0.01)
        return tf.Variable(initial, name=myname, trainable=train)
        
    def gaussian_pdf(self, mean, sample):
        Z = 1.0 / (self.stdd * tf.sqrt(2.0 * np.pi))
        a = -tf.square(sample - mean) / (2.0 * tf.square(self.stdd))
        return Z * tf.exp(a)
    
    def glimpseSensor(self, img, normLoc):
        print ("glimpse one time")
        img=tf.reshape(img,(self.batch_size,self.channels,self.img_size_x,self.img_size_y))
        img=tf.transpose(img,perm=[0,2,3,1])
        loc = tf.round(((normLoc + 1) / 2.0) * [self.img_size_x,self.img_size_y])  # normLoc coordinates are between -1 and 1
        loc = tf.cast(loc, tf.int32)
        # process each image individually
        zooms = []
        for k in range(self.batch_size):
            imgZooms = []
            one_img = img[k,:,:,:]
            max_radius_x = int(math.ceil(self.sensorX/2) * (2 ** (self.depth - 1)))
            offset_x = max_radius_x
            max_radius_y = int(math.ceil(self.sensorY/2) * (2 ** (self.depth - 1)))
            offset_y = max_radius_y
            # pad image with zeros
            one_img = tf.image.pad_to_bounding_box(one_img, offset_x, offset_y, max_radius_x * 2 + self.img_size_x, max_radius_y * 2 + self.img_size_y)
            #print (one_img.get_shape())
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
        # input image
        self.img = tf.placeholder(tf.float32, shape=(None, self.channels, self.time, self.img_size_x, self.img_size_y))
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
        
        #loc_network
        self.location_layer_w=self.weight_variable((self.core_layer_output_w,2),"location_layer_w",True)

        Hidden_layer=[]
        Hidden_layer.append(tf.random_uniform((self.batch_size,self.core_layer_output_w),minval=-0.01, maxval = 0.01))
        Locs=[]
        #Locs.append(tf.random_uniform((self.batch_size, 2), minval=-0.01, maxval=0.01))
        Shirink_edge=min(float(self.sensorX)/self.img_size_x,float(self.sensorY)/self.img_size_y)
        #print (Shirink_edge)
        Locs.append(tf.random_uniform((self.batch_size, 2), minval=(-0.8), maxval=(0.8)))
        count=0
        Location_Prob = []
        for i in range(self.time):
            Img = self.img[:, :, i, :, :]
            for j in range(self.repeat):
                Gli_input = tf.reshape(Img, (-1, Glimpse_input))
                location_use=Locs[count]
                location_use=tf.stop_gradient(location_use)
                Gli_input = self.glimpseSensor(Gli_input, location_use)
                Hidden_layer.append(tf.nn.relu(
                    tf.matmul(Gli_input, self.core_layer_w) + self.core_layer_b + tf.matmul(Hidden_layer[count],
                                                                                            self.transfer_layer_w)))

                loc_mean = tf.matmul(Hidden_layer[-1], self.location_layer_w)
                #loc_mean=tf.stop_gradient(loc_mean)
                loc_std = tf.random_normal([self.batch_size, 2], mean=0, stddev=self.stdd)
                loc = loc_std + loc_mean
                loc = tf.maximum(0.8, tf.minimum(0.8, loc))
                #loc = tf.maximum(-0.01, tf.minimum(0.01, loc))
                #loc=tf.stop_gradient(loc)
                if (count + 1) < self.time * self.repeat:
                    probability=tf.reduce_sum(tf.square(self.gaussian_pdf(loc_mean, loc)),axis=1)/2.0
                    Location_Prob.append(probability)

                Locs.append(loc)
                count = count + 1
        return (Hidden_layer[1:(len(Hidden_layer))], Location_Prob,Locs)

             
        
    def fit(self,train_data,train_label):
        np.random.seed(1337)
        tf.set_random_seed(1337)

        n_samples, self.channels, self.time,self.img_size_x,self.img_size_y= train_data.shape[0:5]



        train_label_mat=[]
        for i in train_label:
            train_label_mat.append([i])
        train_label_mat=np.array(train_label_mat)


        self.n_classes = 1
        Hidden_layer,Pro, self.Location_out=self.build_model()
        assert len(Hidden_layer)==self.time
        assert len(Pro)==self.time-1



        self.y_conv=[]
        self.yy_multi=[]
        for i in range(self.time):
            self.y_conv.append(tf.divide(1.0,tf.minimum(tf.add(1.0,tf.exp(-(tf.matmul(Hidden_layer[i],self.action_layer_w)+self.action_layer_b))),1000.0)))       
            self.yy_multi.append(self.yy)
        with tf.name_scope("kkkkkkkkkkkkkkk"):
            self.y_conv=tf.stack(self.y_conv)
        self.yy_multi=tf.stack(self.yy_multi)


        with tf.Session() as sess:
            #shape=(time, batch , 1)

            cross_entropy = (tf.multiply(self.yy_multi,tf.log(self.y_conv))+tf.multiply((1.0-self.yy_multi),tf.log((1.0-self.y_conv+0.0001))))


            correct_prediction = tf.abs(self.y_conv-self.yy_multi)<0.5



            r_pre = tf.reshape(tf.to_float(correct_prediction),(self.time, self.batch_size,1))
            r=r_pre[1:r_pre.get_shape()[0],:,:]

            Rew_eve=np.zeros([(self.time-1), self.batch_size,1])
            for i in range(self.batch_size):
                '''
                for j in range((self.time-1)):
                    if r[j,i,0]==1.0:
                        for m in range((j+1)):
                            Rew_eve[m,i,0]=Rew_eve[m,i,0]+1
                '''
                if r[(self.time-2),i,0]==1.0:
                    for m in range((self,time-1)):
                            Rew_eve[m,i,0]=Rew_eve[m,i,0]+1

            Rew_eve=tf.stack(Rew_eve)
            Rew_eve=tf.cast(Rew_eve, tf.float32)

            assert Rew_eve.get_shape()[0]==self.time-1
            assert Rew_eve.get_shape()[1]==self.batch_size
            assert Rew_eve.get_shape()[2]==1
            '''
            Decay_reward=np.ones(((self.time-1), self.batch_size,1))
            
            for i in range(self.time):
                for j in range(self.batch_size):
                    Decay_reward[i,j]=[np.exp(-(self.time-i-1.0)/self.time)]
            
            Decay_reward=tf.stack(Decay_reward)
            Decay_reward=tf.cast(Decay_reward, tf.float32)
            assert Decay_reward.get_shape()[0]==self.time-1
            assert Decay_reward.get_shape()[1]==self.batch_size
            assert Decay_reward.get_shape()[2]==1
            
            r=tf.multiply(r,Decay_reward)

            print (tf.stack(Pro).get_shape())
            '''
            log_p = tf.reshape(tf.log(tf.stack(Pro)+1e-10), ((self.time-1),self.batch_size,1))
            #log_p = tf.transpose(log_p, perm=[1,0,2])
            #log_p=tf.reduce_sum(log_p,axis=2)

            
            # combine
            BB=tf.transpose(tf.multiply(Rew_eve,log_p), perm=[1,0,2])
  
            '''
            B=tf.reduce_mean(BB,axis=0)
            B_aver=[]
            for i in range(self.batch_size):
                B_aver.append(B)
            B=tf.stack(B_aver)
            #B=tf.reshape(B,(self.time,self.batch_size,1))
            #shape=(batch ,time,  1)
            Rew=BB-B
            '''
            Rew=BB
            '''
            Reward_decay=[]
            for i in  range(self.time-1):
                Reward_decay_line=[]
                num_count=0
                for j in range(i,(self.time-1)):
                    num_count=num_count+1.0
                    Reward_decay_line.append(Rew[:,j,:]*0.99**(j-i))
                Reward_decay_line=tf.reduce_sum(tf.stack(Reward_decay_line),axis=0)
                assert Reward_decay_line.get_shape()[0]==self.batch_size
                assert Reward_decay_line.get_shape()[1]==1
                Reward_decay.append(Reward_decay_line/num_count*self.time)
            Reward_decay=tf.transpose(tf.stack(Reward_decay),[1,0,2])
            print (Reward_decay.get_shape())

            '''


            cross_entropy=cross_entropy[1:cross_entropy.get_shape()[0],:,:]
            
            cross_eve_before=np.zeros([(self.time-2), self.batch_size,1])
            cross_eve_last=np.ones([1, self.batch_size,1])
            cross_eve=np.concatenate((cross_eve_before,cross_eve_last),axis=0)
            cross_eve=tf.stack(cross_eve)
            cross_eve=tf.cast(cross_eve,tf.float32)
            cross_entropy=tf.multiply(cross_entropy,cross_eve)
            cross_entropy=tf.transpose(cross_entropy,perm=[1,0,2])
            
            J=cross_entropy+Rew 


            J=tf.squeeze(J)

            J = tf.reduce_sum(J, axis=1)
            
 
            J = tf.reduce_mean(J, axis=0)
            

            cost = -J
            var_train=tf.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var_train if 'bias' not in v.name ]) * 0.0001
            cost=cost+lossL2
            
            self.lr=tf.placeholder("float", None)
            trainer = tf.train.AdamOptimizer(self.lr)
            
            gvs = trainer.compute_gradients(cost)




            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -10, 10)
            clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
            train_step = trainer.apply_gradients(clipped_gradients)

            
            
            
            
       
            
        
            
            
            
            correct_prediction = tf.abs(self.y_conv-self.yy_multi)<0.5
            correct_prediction=tf.squeeze(correct_prediction)
            accurac = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),axis=1)
            print (accurac.get_shape())
            accuracy=accurac[self.time-1]
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
            saver = tf.train.Saver() 
            Index = [i for i in range(len(train_data))]
            for i in range(self.epoch):
                '''
                if i%10==0 and i>50:
                    self.learning_rate=self.learning_rate*0.9
                    print (self.learning_rate)
                    if self.learning_rate<0.0000004:
                        self.learning_rate=0.0000004
                '''
                Total_acc=0.0

                random.shuffle(Index)
                for j in range(int(len(train_data) / self.batch_size)):
                    eve_train_data = []
                    eve_train_label_mat = []
                    for mm in Index[j * self.batch_size:(1 + j) * self.batch_size]:
                        eve_train_data.append(train_data[mm])
                        eve_train_label_mat.append(train_label_mat[mm])
                    train_step.run(feed_dict={self.lr: self.learning_rate, self.img: eve_train_data, self.yy: eve_train_label_mat})
                    train_accuracy = (accuracy.eval(feed_dict={self.img: eve_train_data, self.yy: eve_train_label_mat}))
                    Total_acc = Total_acc + train_accuracy
                    '''
                    CE=sess.run(cross_entropy,feed_dict={self.img: eve_train_data, self.yy: eve_train_label_mat})
                    print (CE*30)
                    Re=sess.run(Rew,feed_dict={self.img: eve_train_data, self.yy: eve_train_label_mat})
                    
                    print (Re)
                    '''
                
                print("Epoch %d,training accuracy " % (i + 1))
                print ( Total_acc / (float(len(train_data)/self.batch_size)))
                del eve_train_data,eve_train_label_mat
                gc.collect()
                saver = tf.train.Saver() 
                folder = os.path.dirname(self.checkpoint_path)
                if not gfile.Exists(folder):
                    gfile.MakeDirs(folder)
                #if (i%100==0):
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
            y_pred = tf.argmax(self.y_conv,1)
            for i in range(int(len(test_data)/self.batch_size)):
                X=[]
                for j in range(self.batch_size):
                    X.append(test_data[i*self.batch_size + j])
                y = y_pred.eval(feed_dict={self.img: X})
                if i == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y,y), axis=0)
        return Y[:(Len+1)]

    
    def clean(self):
        tf.reset_default_graph()