# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:07:58 2017

@author: YimingZhao
"""


from modelGlimpseclassifier import Glimpseclassifier
from expConfig import expConfig
from setting import HoldOut
from datasetMNIST import datasetMNIST,datasetMNIST_embed,datasetMNIST3d
from metric import Accuracy
from Configuration import Configuration_Glimpse

AA=Configuration_Glimpse()
num=AA.batch_size

d=datasetMNIST(n_sample=num)
s = HoldOut()
e = [Accuracy()]
m = Glimpseclassifier(1)
path = 'results/test/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
p.run()