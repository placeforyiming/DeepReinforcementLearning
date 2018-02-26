# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:53:15 2017

@author: YimingZhao
"""

from modelRNNclassifier import RNNclassifier
from expConfig import expConfig
from setting import HoldOut
from datasetMNIST import datasetMNIST,datasetMNIST_embed,datasetMNIST3d
from metric import Accuracy
from Configuration import Configuration_RNN

AA=Configuration_RNN()
num=AA.batch_size
d=datasetMNIST(n_sample=num)
s = HoldOut()
e = [Accuracy()]
m = RNNclassifier(1)
path = 'results/test/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
p.run()