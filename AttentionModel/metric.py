from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score

class metric(object):
    __metaclass__ = ABCMeta

    def __init__(self,name,desc):
        self.name = name
        self.desc = desc
        self.values=[]
    
    @abstractmethod
    def compute(self,trueLabels,preLables=[],outputs=[]):
        pass

class Accuracy(metric):
    def __init__(self):
        metric.__init__(self,'accuracy','Accuracy')

    def compute(self,trueLabels,preLabels,outputs=[]):
        if len(trueLabels)!=len(preLabels):
            raise('evaluation[ErrorRate]: dim mismatch of label and prelabel')
            return
        value = accuracy_score(trueLabels, preLabels)
        self.values.append(value)

