import tensorflow as tf
import numpy as np
import os
from experiments.models import *
import gc, re, copy
import pandas as pd
import pickle
import glob
from sklearn.model_selection import train_test_split
## set the project folder to something for saving
project_folder=os.getcwd()+"/"### change to desired path

######################################################################
# Classes for callbacks; early stopping and saving checkpoints
######################################################################
class stop_callback(tf.keras.callbacks.Callback):
    """Class to enable early stopping as we reach a certain training error"""
    def __init__(self, monitor='accuracy', value=0.001, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy')>= self.value): # select the accuracy
            print("\n !!! training error threshold reached, no further training !!!")
            self.model.stop_training = True
            
class fast_checkpoints(tf.keras.callbacks.Callback):
    """Class to have more checkpoints in at the start of training"""
    def __init__(self,checkpoint_path,save_freq):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.save_freq=save_freq
        self.filepath=checkpoint_path
        self.verbose=1
        self.save_best_only=False
        self.save_weights_only=True
    def on_train_batch_begin(self, batch, epoch, logs=None):
         if batch%self.save_freq==0:
            ## Create folder
            weight_path = self.filepath+"/1_"+str(batch)+".ckpt"
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            
            self.model.save_weights(weight_path)
            

def get_learning_rate(task,architecture):
    """Function returning the learning rate depending on architecture and task"""
    if architecture=="lenet":
        lr=0.003
        
    elif architecture=="fc":
        lr=0.03
    else:
        lr=0.0003
    return lr
        
    
def train_posterior(alpha,generator=None, val_generator=None, x_train=[],y_train=[],prior_weights=None,x_test=[],y_test=[],save=True,task=2,binary=False,batch_size=128,architecture="lenet",seed=69105,image_size=32):
        """
        Here we start from a prior, a random prior if alpha=0 or otherwise one which we have trained beforehand.
        From this point we train until we achieve some predefined sample error, saving weights 10 times during the first epoch 
        and then once after each subsequent epoch, Finally we return the final posterior weights
        """
        
        ####################################################################
        # Here we set the path and define the callbacks to save checkpoints 
        ####################################################################
    
        checkpoint_path = project_folder+"posteriors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)
        if binary:
            checkpoint_path = project_folder+"posteriors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)
        ## Create the folder for checkpoints
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        # Create a callback that saves the model's weights every epoch
        if generator is not None:
            checkpoint_freq=np.ceil(generator.__len__())
        else:
            checkpoint_freq=np.ceil(len(x_train)/batch_size)
            
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            save_freq="epoch",
            filepath=checkpoint_path+"/2_{epoch:0d}.ckpt", 
            verbose=1,
            save_best_only=False,
            save_weights_only=True,

        )
        ## callback for saving more frequently during the first epoch
        fast_checkpoint_freq=np.ceil(checkpoint_freq/10)
        fast_cp_callback =fast_checkpoints(checkpoint_path,int(fast_checkpoint_freq))
        #stopping_callback=stop_callback(monitor='val_acc',value=1-epsilon)
        
        ###########################################################
        # Load the prior so that we can begin training from there
        ###########################################################
        model=init_task_model(task=task,binary=binary,architecture=architecture,image_size=image_size)
        
        
        if binary:
            prior_path=project_folder+"priors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)+"/prior.ckpt"
        else:
            prior_path=project_folder+"priors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)+"/prior.ckpt"
        ## choose loss function, optimiser etc.
        lr=get_learning_rate(task,architecture)
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.95),
                      metrics=['accuracy'],)    
        ### load the prior weights if given
        if prior_weights is not None:
            model.set_weights(prior_weights)
        elif(alpha==0):
            ## Create the folder and save the prior
            os.makedirs(os.path.dirname(prior_path), exist_ok=True)
            ### save the rand. init as the prior
            model.save_weights(prior_path)
        else:
            ### load the prior which should already be saved
            if binary:
                model.load_weights(prior_path).expect_partial()
            else:   
                model.load_weights(prior_path).expect_partial()

            
        
        ########################################################################################
        # Load the correct model and learning rate for the given task and compile the model 
        ########################################################################################
        prior_weights=model.get_weights()
        ## redo the model so that we may implement regularisation towards the prior if we want
        M=init_task_model(task=task,binary=binary,architecture=architecture,prior_weights=prior_weights,image_size=image_size) 
        M.set_weights(prior_weights)
        
       
        ## remove previous weight checkpoints if there are any
        files=glob.glob(os.path.join(checkpoint_path+'/*'))
        for file in files:
            os.remove(file)
        

        ###########################################################################
        # train for one epoch with more checkpoints to be able to plot more there
        ###########################################################################
    
        if save:
            CALLBACK=[fast_cp_callback]
        else:
            CALLBACK=[]
        if generator is not None:
            fit_info=model.fit(generator, batch_size=batch_size, epochs=1, callbacks=CALLBACK,
                    validation_data=val_generator,verbose=1, workers=10)
        else:
            fit_info = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, callbacks=CALLBACK,
                    validation_data=(x_test, y_test),verbose=1)
        
        print("-"*40)
        print("Finished training first posterior epoch")
        
        
        
        #######################
        # train for X epochs   
        #######################
        
        if save:
            CALLBACK=[cp_callback]
        else:
            CALLBACK=[]
        if generator is not None:
            fit_info = model.fit(generator,
               batch_size=batch_size,
               epochs=5, 
               callbacks=CALLBACK,
               validation_data=val_generator,
               verbose=0,
                workers=10
                            )
        else:
            fit_info = model.fit(x_train, y_train,
               batch_size=batch_size,
               epochs=5, 
               callbacks=CALLBACK,
               validation_data=(x_test, y_test),
               verbose=0,
                            )
        print("-"*40)
        print("Finished training posterior")
        weights=model.get_weights()
        del model
        return weights
    
  
def train_prior(alpha,total_epochs,generator=None,val_generator=None,x_train=[],y_train=[],x_target=[],y_target=[],save=True,task=2,binary=False,batch_size=128,architecture="lenet",seed=69105,image_size=32):
    """
    This function takes in arguments and then trains a prior for one epoch.
    Then we compute sample and target errors for the prior and save that to a file
    """
    
    ####################################################################
    # Here we set the path and define the callbacks to save checkpoints 
    ####################################################################
    
    if binary:
        checkpoint_path= project_folder+"priors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)
        os.makedirs(os.path.dirname(checkpoint_path)+"/", exist_ok=True) 
    else:
        checkpoint_path = project_folder+"priors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)
        os.makedirs(os.path.dirname(checkpoint_path)+"/", exist_ok=True)
    
    
    ### save checkpoints 10 times over the training of the prior
    if generator is not None:
        checkpoint_freq=np.ceil(generator.__len__()/10)
    else:
        checkpoint_freq=np.ceil(len(x_train)/(batch_size*10))
    
    fast_cp_callback =fast_checkpoints(checkpoint_path,int(checkpoint_freq))
    if save:
            CALLBACK=[fast_cp_callback]
    else:
            CALLBACK=[]
    
    ## remove previous weights if there are any
    files=glob.glob(os.path.join(checkpoint_path+'/*'))
    for file in files:
        os.remove(file)
        
        
    ########################################################################################
    # Load the correct model and learning rate for the given task and compile the model 
    ########################################################################################    
    model=init_task_model(task=task,binary=binary,architecture=architecture,prior_weights=None,image_size=image_size)
    lr=get_learning_rate(task,architecture)
    
    ## choose loss function, optimiser etc.
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
               optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.95),
                      metrics=['accuracy'],)
    
    
    ##############################
    # train prior for one epoch 
    ##############################
    if generator is not None:
        fit_info = model.fit(generator,
           batch_size=batch_size,
           callbacks=CALLBACK,
           epochs=1,
           verbose=0,
           workers=10
                        )
    else:
        x_bound, x_prior, y_bound , y_prior = train_test_split(x_train,y_train,test_size=alpha,random_state=seed)
        fit_info = model.fit(x_bound, y_bound,
               batch_size=batch_size,
               callbacks=CALLBACK,
               epochs=1,
               verbose=0,
                            )
    
    ##############################
    # train prior for the rest
    ##############################
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            save_freq="epoch",
            filepath=checkpoint_path+"/2_{epoch:0d}.ckpt", 
            verbose=1,
            save_best_only=False,
            save_weights_only=True,

        )
    if save:
        CALLBACK=[cp_callback]
        
    if generator is not None:
        fit_info = model.fit(generator,
           batch_size=batch_size,
           callbacks=CALLBACK,
           epochs=total_epochs-1,
           verbose=0,
           workers=10
                        )
    else:
        x_bound, x_prior, y_bound , y_prior = train_test_split(x_train,y_train,test_size=alpha,random_state=seed)
        fit_info = model.fit(x_bound, y_bound,
               batch_size=batch_size,
               callbacks=CALLBACK,
               epochs=total_epochs-1,
               verbose=0,
                            )
    
    print("-"*40)
    print("Finished training prior")
    
    #### save the final prior weights to disk
    if save:
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_weights(checkpoint_path+"/prior.ckpt") 
        
        
        
        
     
    ##########################################################################
    # here we sort the filenames of the prior weights and add them to a list
    ##########################################################################
    checkpoint_names=[]
    
    dirFiles = os.listdir(checkpoint_path) #list of directory files
    
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: #filter out all non weights
        if '.ckpt.index' in files:
            name = re.sub('\.ckpt.index$', '', files)
            if (name[0]=="1") or (name[0]=="2"):
                checkpoint_names.append(name)
        
    checkpoint_names.sort(key=lambda f: int(re.sub('\D', '', f)))
    checkpoint_names.append("prior")    ## add the final weights which has no number
    
    
    weight_updates=[]
    error=[]
    target_error=[]
    
    
    for i in checkpoint_names:
        if i[0:2]=="1_":
            weight_updates.append(int(i[2:]))
        elif i=="prior": 
            if generator is not None:
                weight_updates.append(int(generator.__len__()))
            else: 
                weight_updates.append(int(np.ceil(len(y_train)/batch_size)))
    
    
    ##########################################################################
    # here we load each prior weight and compute the sample and target error
    ##########################################################################
    if save:
    
        for checkpoint in checkpoint_names:

            model=init_task_model(task,binary,architecture=architecture,image_size=image_size)
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                   optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.95),
                      metrics=['accuracy'],)
            model.load_weights(checkpoint_path+"/"+str(checkpoint)+".ckpt").expect_partial()
            if generator is not None:
                target_error.append(1-model.evaluate(val_generator,verbose=0, workers=10)[1])
                error.append(1-model.evaluate(generator,verbose=0, workers=10)[1])
            else:
                target_error.append(1-model.evaluate(x_target,y_target,verbose=0)[1])
                error.append(1-model.evaluate(x_train,y_train,verbose=0)[1])
        print("-"*40)
        print("Finished computing prior sample and target errors")

        ####################################################
        # save all the sample and target errors to a file
        ####################################################

    
        results=pd.DataFrame({
            'weight_updates': [weight_updates],
            'train_error': [error],
            'target_error': [target_error],
            'seed': [seed],
            'image_size': [image_size]
            })
       
        ## Create the folders if they do not exist and save the results
        
        if binary:
            result_path=project_folder+"results/"+"task"+str(task)+"/Binary/"+str(architecture)
            os.makedirs(os.path.dirname(result_path+"/"), exist_ok=True)
        else:
            result_path=project_folder+"results/"+"task"+str(task)+"/"+str(architecture)
            os.makedirs(os.path.dirname(result_path+"/"), exist_ok=True)
            
        ## just create the file and save the results
        with open(result_path+"/"+str(image_size)+"_"+str(int(100*alpha))+"_prior_results.pkl",'wb') as f:
            pickle.dump(results,f)
        f.close()
        weights=model.get_weights()
        del model
    return weights
    
