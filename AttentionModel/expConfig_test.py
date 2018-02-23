import os
from expConfig import expConfig
from setting import CrossValidation
from dataset import Toy
from model import KnnClassifier
from metric import Accuracy


def test_profile():
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
    assert os.path.exists(path)
