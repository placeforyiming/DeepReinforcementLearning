# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:25:47 2017

@author: YimingZhao
"""



class Configuration_FC():
    def __init__(self):
        self.input_layer_input_w=100
        self.input_layer_output_w=256
        self.input_layer_output_b=self.input_layer_output_w
        self.core_layer_input_w=self.input_layer_output_w
        self.core_layer_output_w=128
        self.core_layer_output_b=self.core_layer_output_w
        self.action_layer_input_w=self.core_layer_output_w
        self.action_layer_output_w=10
        self.action_layer_output_b=self.action_layer_output_w
        self.batch_size=32
        self.epoch=1
        self.learning_rate=0.00004
        self.N_front=4
        self.checkpoint_path = './save/FC_model'+str(self.epoch)+'.ckpt'
    def Return(self):
        return (self.input_layer_input_w,self.input_layer_output_w, self.input_layer_output_b, self.core_layer_input_w,self.core_layer_output_w,self.core_layer_output_b,self.action_layer_input_w,self.action_layer_output_w,self.action_layer_output_b,self.batch_size,self.epoch,self.learning_rate,self.checkpoint_path)


class Configuration_CNN(Configuration_FC):
    def __init__(self):
        Configuration_FC.__init__(self)
        self.checkpoint_path = './save/CNN_model'+str(self.N_front)+'.ckpt'
        self.filter_depth=3
        self.filter_height=25
        self.filter_width=15
        self.in_channels=5
        self.out_channels=24
        self.stride=[1,1.,2.,2,1]
       
        
    def Return_CNN(self):
        return (self.filter_depth,self.filter_height,self.filter_width,self.in_channels,self.out_channels,self.stride)
    

class Configuration_RNN(Configuration_FC):
    def __init__(self):
        Configuration_FC.__init__(self)
        self.checkpoint_path = './save/RNN_model'+str(self.N_front)+'.ckpt'
        self.repeat=1
        self.transfer_layer_input_w=self.core_layer_output_w
        
    def Return_RNN(self):
       return (self.repeat,self.transfer_layer_input_w)
   
    

    
class Configuration_RandGlimpse(Configuration_RNN):
    def __init__(self):
        Configuration_RNN.__init__(self)
        self.checkpoint_path = './save/RandGlimpse_model'+str(self.N_front)+'.ckpt'
        self.sensorX=25
        self.sensorY=15
        self.depth=3
        self.hg_size=128
        self.hl_size=128
        self.glimpse_size=256
    def Return_RandGlimpse(self):
        return (self.sensorX,self.sensorY,self.depth,self.hg_size,self.hl_size,self.glimpse_size)
         
    
    
class Configuration_Glimpse(Configuration_RandGlimpse):
    def __init__(self):
        Configuration_RandGlimpse.__init__(self)
        self.checkpoint_path = './save/Glimpse_model'+str(self.N_front)+'.ckpt'
        self.stdd=0.12
        self.reward_ratio=1
        self.epoch=1000
        
    def Return_Glimpse(self):
        return (self.stdd,self.reward_ratio)
    
    
class Configuration_3DGlimpse(Configuration_Glimpse):
    def __init__(self):
        Configuration_Glimpse.__init__(self)
        self.checkpoint_path = './save/3DGlimpse_model'+str(self.N_front)+'.ckpt'
        self.repeat=5
        self.sensorTime=8
    
    def Return_3DGlimpse(self):
        return (self.repeat,self.sensorTime)