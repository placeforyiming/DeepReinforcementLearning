from abc import ABCMeta, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class model(object):
    __metaclass__ = ABCMeta
    def __init__(self,name,desc=""):
        self.name = name
        self.desc = desc

    # training
    @abstractmethod
    def fit(self,train_data,train_label):
        return
    
    # predicting 
    @abstractmethod
    def inference(self,test_data):
        return
    
    #clean the model before saving the result (for example: delete parameters with large sizes)
    def clean(self):
        return 



class KnnClassifier(model):
    def __init__(self,n_neighbors = 1):
        model.__init__(self,'knn','k nearest neighbor classifier')
        self.n_neighbors = n_neighbors
        self.clf=None

    def fit(self,train_data, train_label):
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.clf.fit(train_data, train_label)

    def inference(self,test_data):
        return self.clf.predict(test_data)

    
    
class LogisticRegressionClassifier(model):
    def __init__(self,C=1.0,n_jobs=1):
        model.__init__(self,'lrc','logistic regression classifier')
        self.C = C
        self.clf=None
        self.n_jobs=n_jobs

    def fit(self,train_data, train_label):
        self.clf = LogisticRegression(C=self.C,n_jobs=self.n_jobs)
        self.clf.fit(train_data, train_label)

    def inference(self,test_data):
        return self.clf.predict(test_data)
    
    
    
