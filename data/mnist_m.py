import pandas as pd
import numpy as np
def load_mnistm(y_train,y_test): ## take in labels from MNIST as they are the same
    """
    Loads the mnist-m data from a .pkl file
    """


    M = pd.read_pickle('mnistm_data.pkl') ### substitute with MNIST-M file
    x_train_m =M['train']
    x_test_m =M['test']
    y_train_m=y_train
    y_test_m =y_test
    x_train_m = np.pad(x_train_m,((0,0),(2,2),(2,2),(0,0))) #padding to make images 32x32 and not 28x28
    x_test_m = np.pad(x_test_m,((0,0),(2,2),(2,2),(0,0)))

    x_train_m = x_train_m.astype('float32')
    x_test_m = x_test_m.astype('float32')
    ## normalising to unit variance
    sigma=np.std(x_train_m)
    x_train_m /= sigma 
    x_test_m /= sigma

    ## mean subtraction
    mu=np.mean(x_train_m)
    x_train_m -= mu
    x_test_m -= mu

    print('mean, variance', mu, sigma)
    print("---------------Load MNIST-M----------------")
    print('Training set', x_train_m.shape, y_train_m.shape)
    print('Test set', x_test_m.shape, y_test_m.shape)
    
    return x_train_m, y_train_m, x_test_m, y_test_m
