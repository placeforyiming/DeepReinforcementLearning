import sklearn
import tensorflow as tf
import keras

def test_sklearn_version():
    assert sklearn.__version__ == '0.18.1'

def test_tensorflow_version():
    assert tf.__version__=='1.0.1'

def test_keras_version():
    assert keras.__version__=='2.0.1'
