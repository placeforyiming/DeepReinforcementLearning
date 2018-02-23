from dataset import Toy

def test_toydataset():
    d = Toy()
    X, y, X2,y2 = d.load()
    assert X.shape == (100,2)
    assert len(y) == 100