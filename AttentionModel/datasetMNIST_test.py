from datasetMNIST import *

def test_load_mnist():
    d = datasetMNIST()
    x1,y1,x2,y2 = d.load()
    assert x1.shape == (60000,1,1,28,28)
    assert len(y1) == 60000
    assert x2.shape == (10000,1,1,28,28)
    assert len(y2) == 10000


def test_load_mnist_embed():
    d = datasetMNIST_embed()
    x1,y1,x2,y2 = d.load()
    assert x1.shape == (60000,1,1,100,100)
    assert len(y1) == 60000
    assert x2.shape == (10000, 1,1,100,100)
    assert len(y2) == 10000

def test_load_mnist_3d():
    d = datasetMNIST3d()
    x1,y1,x2,y2 = d.load()
    assert x1.shape == (60000,4,3,28,28)
    assert len(y1) == 60000
    assert x2.shape == (10000,4,3,28,28)
    assert len(y2) == 10000
 
    
    
    
