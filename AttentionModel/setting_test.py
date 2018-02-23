from setting import *
from dataset import Toy
from model import KnnClassifier
from metric import Accuracy
from numpy.testing import assert_almost_equal


def test_cv():
    m = KnnClassifier()
    d = Toy() 
    e = Accuracy()
    s = CrossValidation()
    s.setup(d,m,[e])
    s.run()
    assert len(s.metrics[0].values) == 5
    assert_almost_equal(s.metrics[0].values[2], 0.8) 

def test_holdout():
    m = KnnClassifier(n_neighbors=10)
    d = Toy() 
    e = Accuracy()
    s = HoldOut()
    s.setup(d,m,[e])
    s.run()
    assert len(s.metrics[0].values) == 1
    assert_almost_equal(s.metrics[0].values[0], 0.97) 
