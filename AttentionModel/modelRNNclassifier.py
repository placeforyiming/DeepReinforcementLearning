# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:08:51 2017

@author: YimingZhao
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:27:13 2017

@author: yimingzhao
"""
import random
import tensorflow as tf
from model import model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os
from tensorflow.python.platform import gfile
from Configuration import Configuration_RNN
from random import randint

class RNNclassifier(model):
    def __init__(self,isTest):
        model.__init__(self,'RNN','RNN network')
        Conf=Configuration_RNN()
        Conf.Return()
        self.input_layer_input_w,self.input_layer_output_w, self.input_layer_output_b,self.core_layer_input_w,self.core_layer_output_w,self.core_layer_output_b,self.action_layer_input_w,self.action_layer_output_w,self.action_layer_output_b,self.batch_size,self.epoch,self.learning_rate,self.checkpoint_path=Conf.Return()
        self.repeat,self.transfer_layer_input_w=Conf.Return_RNN()
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
        self.core_layer_input_w=self.channels*self.img_size_x*self.img_size_y 
        self.action_layer_output_w=self.n_classes  
        self.action_layer_output_b=self.n_classes
        #on_hot class number
        self.yy = tf.placeholder(tf.float32, shape=(None, self.n_classes))
        
         
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
        count=0
        for i in range(self.time):
            Img=self.img[:,:,i,:,:]
            for j in range(self.repeat):
                RNN_input=tf.reshape(Img,(-1,self.core_layer_input_w))
                Hidden_layer.append(tf.nn.relu(tf.matmul(RNN_input,self.core_layer_w)+self.core_layer_b+tf.matmul(Hidden_layer[count],self.transfer_layer_w)))
                count=count+1
        return Hidden_layer[-1]        
        
    def fit(self,train_data,train_label):
        np.random.seed(1337)
        tf.set_random_seed(1337)
        n_samples, self.channels, self.time,self.img_size_x,self.img_size_y= train_data.shape[0:5]
        
        '''

        self.lb = LabelBinarizer()
        train_label_mat = np.array(self.lb.fit_transform(train_label))
        self.n_classes = train_label_mat.shape[1]
        Hidden_layer=self.build_model()
        self.y_conv =tf.nn.softmax(tf.matmul(Hidden_layer,self.action_layer_w)+self.action_layer_b)






        '''



        '''
        with tf.Session() as sess:
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_conv, labels=self.yy))
            correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.yy,1))
    
            trainer=tf.train.AdamOptimizer(self.learning_rate)
            Grads_and_Vars = trainer.compute_gradients(cross_entropy)
            train_step=trainer.apply_gradients(grads_and_vars=Grads_and_Vars)
            correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.yy,1))



        '''


        train_label_mat=[]
        for i in train_label:
            train_label_mat.append([i])
        train_label_mat=np.array(train_label_mat)
        print (tf.shape(train_label_mat))

        self.n_classes = 1
        Hidden_layer=self.build_model()
        #with tf.name_scope("kkkkkkkkkkkkkkk"):
        self.y_conv =tf.divide(1.0,tf.add(1.0,tf.exp(-(tf.matmul(Hidden_layer,self.action_layer_w)+self.action_layer_b))))

        with tf.Session() as sess:
            cross_entropy = tf.reduce_mean(tf.multiply(self.yy,tf.log(self.y_conv+0.0001))+tf.multiply((1.0-self.yy),tf.log((0.0001+1.0-self.y_conv))))


            var_train=tf.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var_train if 'bias' not in v.name ]) * 0.0001
            cross_entropy=cross_entropy+lossL2
            
            #self.lr=tf.placeholder("float", None)
            trainer = tf.train.AdamOptimizer(self.learning_rate)
            
            gvs = trainer.compute_gradients(cross_entropy)




            def ClipIfNotNone(grad):
                if grad is None:
                    return grad
                return tf.clip_by_value(grad, -1, 1)
            clipped_gradients = [(ClipIfNotNone(grad), var) for grad, var in gvs]
            train_step = trainer.apply_gradients(clipped_gradients)



            correct_prediction = tf.abs(self.y_conv-self.yy)<0.5
        













            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            sess.run(tf.global_variables_initializer())
            
            train_data = train_data.tolist()
            train_label = train_label.tolist()
            train_label_mat = train_label_mat.tolist()
            if len(train_data) % self.batch_size != 0:
                for i in range(len(train_data) % self.batch_size):
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
                    
                    Output=self.y_conv.eval(feed_dict={self.img: eve_train_data, self.yy: eve_train_label_mat})
                  

                    Total_acc = Total_acc + train_accuracy
                print("Epoch %d,training accuracy %g" % (i + 1, Total_acc / (float(len(train_data)/self.batch_size))))
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
        
                