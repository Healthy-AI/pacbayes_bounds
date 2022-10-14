import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import numpy as np
def load_mnist(task):
    """
    Imports the MNIST dataset from the tf datasets
    """
    
    num_classes = 10
    (x_train, lbl_train), (x_test, lbl_test) = mnist.load_data()
    x_train = np.pad(x_train,((0,0),(2,2),(2,2))) #padding to make images 32x32 and not 28x28
    x_test = np.pad(x_test,((0,0),(2,2),(2,2))) 

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    ## normalising to unit variance
    sigma=np.std(x_train)
    x_train /= sigma 
    x_test /= sigma

    ## mean subtraction
    mu=np.mean(x_train)
    x_train -= mu
    x_test -= mu
    ## make labels into categorical classes
    y_train = keras.utils.to_categorical(lbl_train, num_classes)
    y_test = keras.utils.to_categorical(lbl_test, num_classes)


    x_train=np.expand_dims(x_train,3)
    x_test=np.expand_dims(x_test,3)

    if task==4:
        pass
    else:
        ### make mnist into 3 channels
        x_train=np.concatenate((x_train,)*3, axis=-1)
        x_test=np.concatenate((x_test,)*3, axis=-1)
    print('mean, variance', mu, sigma)
    print("---------------Load MNIST----------------")
    print('Training set', x_train.shape, y_train.shape)
    print('Test set', x_test.shape, y_test.shape)
    print("\n")
    return x_train, y_train, x_test, y_test