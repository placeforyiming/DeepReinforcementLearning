import os
from results import results
from nose.tools import assert_almost_equals
from expConfig import expConfig
from setting import CrossValidation
from dataset import Toy
from model import KnnClassifier
from metric import Accuracy

def test_results():
    path = 'results/test/test.pkl'
    m = KnnClassifier()
    d = Toy() 
    s = CrossValidation()
    e = [Accuracy()]
    p = expConfig(dataset=d,
                  setting=s,
                  model=m,
                  metrics=e,
                  resultPath=path)
    p.skip_if_file_exist = False
    p.run()

    r = results(root_dir='results/test')
    x = r.load()
    assert_almost_equals(x[0].metrics[0].values[2], 0.8) 
