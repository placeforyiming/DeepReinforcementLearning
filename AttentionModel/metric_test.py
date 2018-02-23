from metric import Accuracy

def test_accuracy():
    e = Accuracy()
    e.compute([1,-1,1,-1],[1,1,1,1])
    assert e.values[0]==0.5
    e.compute([1,-1,1,-1],[1,1,1,-1])
    assert e.values[1]==0.75
    assert len(e.values) == 2

