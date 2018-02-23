import pickle
import os
from tensorflow.python.platform import gfile

class expConfig:
    def __init__(self, dataset, setting, model, metrics ,resultPath, skip_if_file_exist = True):
        self.dataset = dataset
        self.setting = setting
        self.model = model
        self.metrics = metrics
        self.resultPath = resultPath
        self.skip_if_file_exist = skip_if_file_exist
        
    def run(self):
        if self.skip_if_file_exist and os.path.isfile(self.resultPath):
            print ("file exist, experiment skipped")
            return
        self.setting.setup(dataset=self.dataset,
                             model=self.model,
                             metrics=self.metrics)
        self.setting.run()
        #self.save_result(self.setting,self.resultPath)

    def save_result(self,result,filename):
        result.model.clean()
        folder = os.path.dirname(filename)
        if not gfile.Exists(folder):
            gfile.MakeDirs(folder)
        with open(filename, 'wb') as out_file:
            pickle.dump(result, out_file, pickle.HIGHEST_PROTOCOL)