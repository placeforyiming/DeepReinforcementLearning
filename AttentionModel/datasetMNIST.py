from dataset import dataset
from keras.datasets import mnist
import numpy as np

# tensor shape: batch_size, 1 channel, 1 time slices, 28, 28
class datasetMNIST(dataset):
    def __init__(self, n_sample=-1):
        dataset.__init__(self,'mnist','MNIST dataset')
        self.n_sample = n_sample

    def load(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        # batch_size, channel, time, x, y
        X_train = X_train.reshape((n_train,1,1,28,28))
        X_test = X_test.reshape((n_test,1,1,28,28))
        if self.n_sample>0:
            X_train = X_train[:self.n_sample]
            y_train = y_train[:self.n_sample]
            X_test = X_test[:self.n_sample]
            y_test = y_test[:self.n_sample]



        return  X_train, y_train,X_test, y_test

# tensor shape: batch_size, 1 channel, 1 time slices, 100, 100
class datasetMNIST_embed(dataset):
    def __init__(self, 
                 embed_dim = [100,100],
                 n_sample=-1,
                 random_seed=1337
                ):
        dataset.__init__(self,'mnistem','MNIST dataset with images embedded into a larger background')
        self.embed_dim=embed_dim
        self.random_seed = random_seed
        self.n_sample=n_sample
        self.image_shape=embed_dim 

    def load(self):
        np.random.seed(self.random_seed)  # for reproducibility
        X_train, y_train, X_test, y_test = datasetMNIST(n_sample=self.n_sample).load()

        X_train=self.embed_image(X_train)
        X_test=self.embed_image(X_test)



        return  X_train, y_train,X_test, y_test

    
    def embed_image(self, data):
        n_data = data.shape[0]
        lx = np.random.randint(0, self.embed_dim[0] - 28,size=n_data)
        ly = np.random.randint(0, self.embed_dim[1] - 28,size=n_data)
        embed_data=np.zeros((n_data,1,1,self.embed_dim[0],self.embed_dim[1]))
        for i in range(n_data):
            embed_data[i,0,0,lx[i]:lx[i]+28,ly[i]:ly[i]+28] = data[i]
        return embed_data

# tensor shape: batch_size, 4 channels, 3 time slices, 28, 28
class datasetMNIST3d(dataset):
    def __init__(self, 
                 n_channels=4,
                 n_times = 3,
                 n_sample=-1
                ):
        dataset.__init__(self,'mnist3d','MNIST dataset 3d extention')
        self.n_channels=n_channels
        self.n_times=n_times
        self.n_sample=n_sample


    def load(self):
        X_train, y_train, X_test, y_test = datasetMNIST(n_sample=self.n_sample).load()

        X_train=np.tile(X_train, (1, self.n_channels, self.n_times, 1, 1))

        X_test=np.tile(X_test, (1, self.n_channels, self.n_times, 1, 1))



        return X_train, y_train,X_test, y_test

