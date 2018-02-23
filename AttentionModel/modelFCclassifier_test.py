# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:17:45 2017

@author: yimingzhao
"""
from modelFCclassifier import FCclassifier
from expConfig import expConfig
from setting import HoldOut
from datasetMNIST import datasetMNIST,datasetMNIST_embed,datasetMNIST3d
from metric import Accuracy
from Configuration import Configuration_FC

AA=Configuration_FC()
num=AA.batch_size
d=datasetMNIST(n_sample=num)
s = HoldOut()
e = [Accuracy()]
m = FCclassifier(1)
path = 'results/test/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
p.run()