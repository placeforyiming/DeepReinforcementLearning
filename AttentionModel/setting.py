from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.model_selection import KFold

class setting(object):
    __metaclass__ = ABCMeta

    def __init__(self,name,desc):
        self.name = name
        self.desc = desc
        self.model = None
        self.dataset = None
        self.metrics = None

    def setup(self, dataset,model,metrics):
        self.model=model
        self.metrics=metrics
        self.dataset=dataset

    @abstractmethod
    def run(self):
        return

class CrossValidation(setting):
    def __init__(self,n_fold=5,random_seed=513):
        setting.__init__(self,'cv','Cross Validation')
        self.n_fold=n_fold
        self.random_seed=random_seed
    
    def run(self):
        kf = KFold(n_splits=self.n_fold,shuffle=True,random_state=self.random_seed)
        X_train, y_train, X_test, y_test = self.dataset.load()
        data = np.concatenate((X_train, X_test), axis=0)
        label = np.concatenate((y_train, y_test), axis=0)
        kf.get_n_splits(data)
        for train_index, test_index in kf.split(data): 
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            self.model.fit(X_train,y_train)
            y_predict = self.model.inference(X_test)
            for e in self.metrics:
                e.compute(y_test,y_predict)
                
class HoldOut(setting):
    def __init__(self):
        setting.__init__(self,'holdout','Hold Out Train Test Split')
        pass
    
    def run(self):
        X_train, y_train, X_test, y_test = self.dataset.load()
        self.model.fit(X_train,y_train)
        y_predict = self.model.inference(X_test)

        #for e in self.metrics:
        #   e.compute(y_test,y_predict)