import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Model
import os

def init_resnet_model(binary=True,task=2,image_size=124):
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    import numpy as np

    if task==4:
        base_model = ResNet50V2(weights='imagenet',include_top=False,input_shape=(image_size,image_size,1),pooling='avg')
    else:
        base_model = ResNet50V2(weights='imagenet',include_top=False,input_shape=(image_size,image_size,3),pooling='avg')
    # Freeze base model
    base_model.trainable = False
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    if binary:
        predictions = Dense(2, activation='softmax')(x)
    else:
        predictions = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def init_fc_model(binary=True):
    """
    Fully connected model, similar to rivasplata et al., dziugaite et al. etc.
    """
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential()
    model.add(Dense(1024,input_shape=(32,32,3), activation = 'relu'))
    model.add(Dense(600, activation = 'relu'))
    model.add(Dense(600, activation = 'relu'))
    model.add(Flatten())
    if binary:
        model.add(Dense(2, activation = 'softmax'))
    else:
        model.add(Dense(10, activation = 'softmax'))
    return model
def init_lr_model(flattened_size=3072,binary=True):
    """
    Logistic regression model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer,Dense
    model = Sequential()
    model.add(Dense(flattened_size,input_shape=(32,32,3),activation='relu'))
    model.add(Flatten())
    if binary:
        model.add(Dense(2,activation="softmax"))
    else:
        model.add(Dense(10, activation='softmax'))
    return model
    
## implement LeNet-5-like architecture
def init_svhn_model(binary):
    """  
    Model used in Dziugaite for training on SVHN
     SGD with:
         momentum: 0.9
         weight_decay: 0.0005
         dropout
         l2 regularization
    """
    model = Sequential()
    model.add(Conv2D(64,(5,5),strides=(1,1), activation='relu',input_shape=(32,32,3),kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) ## 6 5x5 conv kernels
    model.add(Dropout(0.9))
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(64,(5,5),strides=(1,1), activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) ## 16 5x5 conv kernels
    model.add(Dropout(0.75))
    model.add(AveragePooling2D(pool_size=(2, 2),strides=(2, 2)))
    model.add(Conv2D(128,(5,5),strides=(1,1), activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005)))
    model.add(Dropout(0.75))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(3072, activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005)))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005)))
    model.add(Dropout(0.5))
    if binary:
        model.add(Dense(10, activation='softmax',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) # output layer
    else:
        model.add(Dense(10, activation='softmax',kernel_constraint=max_norm(4), bias_constraint=max_norm(4),kernel_regularizer=L2(0.0005), bias_regularizer=L2(0.0005))) # output layer
    
    return model

# ### regularizer toward some prior weights
@tf.keras.utils.register_keras_serializable(package='Custom', name='l2-prior')
class l2_prior_reg(tf.keras.regularizers.Regularizer):
  def __init__(self, l2=0.001,prior_weight_matrix=None):
    self.l2 = l2
    self.prior_weights=prior_weight_matrix

  def __call__(self, x):
    prior=tf.convert_to_tensor(self.prior_weights)
    print("prior is: ",prior.shape)
    print("weights are: ",x.shape)
    return self.l2 * tf.math.reduce_sum(tf.math.square(x-prior))
  def get_config(self):
    return {'l2': float(self.l2), 'prior_weight_matrix': self.prior_weights}
    
def init_mnist_model(binary,prior_weights=None):
    """
    LeNet-5 type model 
    """
    if prior_weights==None:
        model = Sequential()
        model.add(Conv2D(32,(5,5),strides=(1,1), activation='relu',input_shape=(32,32,3))) ## 6 5x5 conv kernels
        model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
        model.add(Conv2D(48,(5,5),strides=(1,1), activation='relu')) ## 16 5x5 conv kernels
        model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        if binary:
            model.add(Dense(2, activation='softmax')) # output layer
        else:
            model.add(Dense(10, activation='softmax')) # output layer
    else:
        model = Sequential()
        model.add(Conv2D(32,(5,5),strides=(1,1), activation='relu',input_shape=(32,32,3),kernel_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[0]),bias_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[1]))) ## 6 5x5 conv kernels
        model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
        model.add(Conv2D(48,(5,5),strides=(1,1), activation='relu',kernel_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[2]),bias_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[3]))) ## 16 5x5 conv kernels
        model.add(AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu',kernel_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[4]),bias_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[5])))
        model.add(Dense(100, activation='relu',kernel_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[6]),bias_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[7])))
        if binary:
            model.add(Dense(2, activation='softmax',kernel_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[8]),bias_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[9]))) # output layer
        else:
            model.add(Dense(10, activation='softmax',kernel_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[8]),bias_regularizer=l2_prior_reg(prior_weight_matrix=prior_weights[9]))) # output layer
    return model

def init_task_model(task=2,binary=True,architecture="lenet",prior_weights=None,image_size=32): 
    """
     Function that takes in the task number and architecture
     
     It returns the model which fits the task
    """
    arch=architecture
    if arch not in ["lr","lenet","fc","resnet"]:
        raise Exception('Architecture '+arch+' not implemented/tested')
    if prior_weights==None:
        ## Load the model w/o regularization
        if arch=="lr":
            model=init_lr_model(binary)
        elif arch=="lenet":
            model=init_mnist_model(binary)
        elif arch=="fc":
            model=init_fc_model(binary)
        else:
            model=init_resnet_model(binary,image_size=image_size)
    else:
        ## Load the model w/ regularization
        if arch=="lr":
            model=init_lr_model(binary)
        elif arch=="lenet":
            model=init_mnist_model(binary,prior_weights)
        elif arch=="fc":
            model=init_fc_model(binary)
        else:
            model=init_resnet_model(binary,image_size=image_size)
    return model
