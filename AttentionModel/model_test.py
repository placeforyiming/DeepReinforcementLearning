from model import *

def test_knn():
    c = KnnClassifier()
    X = [[0], [1], [2], [3]]
    y = [2, 2, 3, 3]
    c.fit(X,y)
    print 
    assert c.inference([[1.1]]) == 2
    assert c.inference([[1.9]]) == 3
    
def test_lrc():
    c = LogisticRegressionClassifier()
    X = [[0], [1], [2], [3]]
    y = [2, 2, 3, 3]
    c.fit(X,y)
    assert c.inference([[0.2]]) == 2
    assert c.inference([[1.1]]) == 3
    assert c.inference([[1.9]]) == 3
