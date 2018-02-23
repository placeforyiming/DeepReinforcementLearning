import os
from expConfig import expConfig
from setting import HoldOut
from datasetMNIST import datasetMNIST,datasetMNIST_embed,datasetMNIST3d

from metric import Accuracy
from model import LogisticRegressionClassifier
from modelFCclassifier import FCclassifier
'''
from modelCNNclassifier import CNNclassifier
from modelRNNclassifier import RNNclassifier
from modelRandGlimpseclassifier import RandGlimpseclassifier
from modelGlimpseclassifier import Glimpseclassifier
from modelthreeDGlimpseclassifier import threeDGlimpseclassifier
'''
#-------------------------------------------------------------------
if (1):
    #0.99
    print ('Start FC Model')
    d = datasetMNIST()
    s = HoldOut()
    e = [Accuracy()]
    m = FCclassifier(isTest=0)
    path = 'results/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
    p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
    p.run()


if (0):
    #1.0
    print ('Start CNN Model')
    d = datasetMNIST_embed()
    s = HoldOut()
    e = [Accuracy()]
    m = CNNclassifier(isTest=0)
    path = 'results/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
    p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
    p.run()

if (0):
    #0.525
    print ('Start RandGlimpse Model')
    d = datasetMNIST_embed()
    s = HoldOut()
    e = [Accuracy()]
    m = RandGlimpseclassifier(isTest=0)
    path = 'results/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
    p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
    p.run()





if (0):
    # 0.985
    print ('Start RNN Model')
    d = datasetMNIST_embed()
    s = HoldOut()
    e = [Accuracy()]
    m = RNNclassifier(isTest=0)
    path = 'results/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
    p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
    p.run()

if (0):
    #0.5
    print ('Start Glimpse Model')
    d = datasetMNIST_embed()
    s = HoldOut()
    e = [Accuracy()]
    m = Glimpseclassifier(isTest=0)
    path = 'results/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
    p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
    p.run()

    
if (0):
    
    print ('Start 3DGlimpse Model')
    d = datasetMNIST_embed()
    s = HoldOut()
    e = [Accuracy()]
    m = threeDGlimpseclassifier(isTest=0)
    path = 'results/'+s.name+'/'+d.name+'/'+m.name+'.pkl'
    p = expConfig(dataset=d,
                setting=s,
                model=m,
                metrics=e,
                resultPath=path)
    p.run()