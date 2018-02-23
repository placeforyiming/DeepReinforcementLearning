# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:51:11 2017

@author: YimingZhao
"""

from modelCNNclassifier import CNNclassifier
from expConfig import expConfig
from setting import HoldOut
from datasetMNIST import datasetMNIST,datasetMNIST_embed,datasetMNIST3d
from metric import Accuracy
from Configuration import Configuration_CNN

AA=Configuration_CNN()
num=AA.batch_size
d=datasetMNIST(n_sample=num)
s = HoldOut()
e = [Accuracy()]
m = CNNclassifier(1)
path = 'results/test/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
p.run()