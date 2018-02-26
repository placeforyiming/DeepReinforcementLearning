# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:22:06 2017

@author: YimingZhao
"""

from modelRandGlimpseclassifier import RandGlimpseclassifier
from expConfig import expConfig
from setting import HoldOut
from datasetMNIST import datasetMNIST,datasetMNIST_embed,datasetMNIST3d
from metric import Accuracy
from Configuration import Configuration_RandGlimpse

AA=Configuration_RandGlimpse()
num=AA.batch_size

d=datasetMNIST(n_sample=num)
s = HoldOut()
e = [Accuracy()]
m = RandGlimpseclassifier(1)
path = 'results/test/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
p.run()
