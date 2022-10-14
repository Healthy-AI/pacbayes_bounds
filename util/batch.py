import os, re

project_folder=os.getcwd()+"/" # change as desired
def get_job_args(task, bound='germain', alpha=0.1, sigma=[3,2], binary=False, n_classifiers=4, architecture="lenet",seed=69105,image_size=32,batch_size=128):
    """
    This function provides the prerequisite arguments which we use to compute our bound
    """
 
    if binary:
        prior_path=project_folder+"priors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)+"/prior.ckpt"
        result_path=project_folder+"results/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"
    else:# +str(int(1000*epsilon))+"_"
        prior_path=project_folder+"priors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)+"/prior.ckpt"
        result_path=project_folder+"results/"+"task"+str(task)+"/"+str(architecture)+"/"+str(int(100*alpha))+"_"
     
    posterior_paths = posterior_checkpoints(task, alpha, binary,architecture,seed,image_size=image_size)
    #### iterate over the list of posterior paths and construct argument list
    arg_list = []
    for post in posterior_paths: 
        args = {
            'task': task, 
            'prior_path': prior_path, 
            'posterior_path': post,
            'bound': bound, 
            'alpha': alpha,
            'sigma': sigma, 
            #'epsilon': epsilon, 
            'binary': binary,
            'n_classifiers': n_classifiers,
            'image_size': image_size,
            'batch_size': batch_size
        }
        arg_list.append(args)
        
    return arg_list
    
def posterior_checkpoints(task, alpha, binary=False,architecture="lenet",seed=69105,image_size=32):
    """
    Since we have saved the posterior weights with the number of weight updates in their name,
    we parse the filenames and sort them in numerical order and then return the list of ordered paths
    """
    if binary:
        base_path=project_folder+"posteriors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)
    else:
        base_path=project_folder+"posteriors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha)+"_"+str(seed))
    ##################################################################################################
    # parse filenames into an ordered list, first the ones with a 1 in them and then the ones with a 2
    ##################################################################################################  
    list1=[]
    list2=[]
    dirFiles = os.listdir(base_path) #list of directory files
    ## remove the ckpt.index and sort so that we get the epochs that are in the directory
    for files in dirFiles: 
        if '.ckpt.index' in files:
            name = re.sub('\.ckpt.index$', '', files)
            ### if it has a one it goes in one list and if it starts with a two it goes in the other
            if (name[0]=="1"):
                list1.append(name)
            elif (name[0]=="2"):
                list2.append(name)
                
    list1.sort(key=lambda f: int(re.sub('\D', '', f)))
    num_batchweights=len(list1)
    list2.sort(key=lambda f: int(re.sub('\D', '', f)))
    list1.extend(list2)
    Ws=list1
        
    path=project_folder+"posteriors/"+"task"+str(task)+"/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)+"/"
    if binary:
        path=project_folder+"posteriors/"+"task"+str(task)+"/Binary/"+str(architecture)+"/"+str(image_size)+"_"+str(int(100*alpha))+"_"+str(seed)+"/"
    
    posterior_paths = [os.path.join(path, str(checkpoint)+".ckpt") for checkpoint in Ws]
    
    return posterior_paths