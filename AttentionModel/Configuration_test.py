# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:34:48 2017

@author: YimingZhao
"""

from Configuration import Configuration_FC
from Configuration import Configuration_CNN
from Configuration import Configuration_RNN
from Configuration import Configuration_RandGlimpse
from Configuration import Configuration_Glimpse
from Configuration import Configuration_3DGlimpse

AA=Configuration_FC()
assert len(AA.Return())==13
          
          
AA=Configuration_CNN()
assert len(AA.Return())==13
assert len(AA.Return_CNN())==6
          



AA=Configuration_RNN()
assert len(AA.Return())==13
assert len(AA.Return_RNN())==2
          
          
AA=Configuration_RandGlimpse()
assert len(AA.Return())==13
assert len(AA.Return_RNN())==2
assert len(AA.Return_RandGlimpse())==6
          
AA=Configuration_Glimpse()
assert len(AA.Return())==13
assert len(AA.Return_RNN())==2
assert len(AA.Return_RandGlimpse())==6
assert len(AA.Return_Glimpse())==2
          
AA=Configuration_3DGlimpse()
assert len(AA.Return())==13
assert len(AA.Return_RNN())==2
assert len(AA.Return_RandGlimpse())==6
assert len(AA.Return_Glimpse())==2
         
         