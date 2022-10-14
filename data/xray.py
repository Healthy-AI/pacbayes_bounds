import tensorflow as tf
import numpy as np
import collections
import pandas as pd
import os,sys
def tf_load_image(filename):
    """
    Load in image so we can handle it
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image=tf.cast(image, tf.float32)
    return image
def tf_read_and_resize_image(filename,img_size):
    """
    takes file path and image size and resizes the image
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image=tf.cast(image, tf.float32)
    image = tf.image.resize(image, (img_size, img_size))
    return image

def load_resize_and_save(chexpert=False,img_size=32):
    """
    loads the xray datasets from the raw images and then resizes to desired size and saves them in a new directory
    """

    if chexpert:
        data_path="/home/adam/Code/Datasets/chexpert/" ### change to where you keep your data
        _DATA_DIR = "/home/adam/Code/Datasets/chexpert/CheXpert-v1.0-small" ### where is the source data?
        _TRAIN_DIR = os.path.join(_DATA_DIR, "train")
        _VALIDATION_DIR = os.path.join(_DATA_DIR, "valid")
        _TRAIN_LABELS_FNAME = os.path.join(_DATA_DIR, "train.csv")
        _VALIDATION_LABELS_FNAME = os.path.join(_DATA_DIR, "valid.csv")
        
        filenames,labels=load_from_csv(data_path,_TRAIN_LABELS_FNAME)
        
        filenames_2,labels_2=load_from_csv(data_path,_VALIDATION_LABELS_FNAME)

        filenames.extend(filenames_2)
        labels.extend(labels_2)
     
        new_path=data_path+"resized"+str(img_size)+"chex"
        ### checks if the filename has a jpg or not and saves it to the new folder
        for file in filenames:
            img=np.array(tf_read_and_resize_image(file,img_size))
            
            A=file.split('/')[-4:]
            path=""
            for a in A:
                if a[-4:]==".jpg":
                    os.makedirs(new_path+path, exist_ok=True)
                    
                path+="/"+a
                
            output_path=new_path+path
            
            tf.keras.preprocessing.image.save_img(output_path,img)
    else:
        data_path="/home/adam/Code/Datasets/chestXray14/"### change to where you keep your data

        dirs = [l for l in os.listdir(data_path+"images/") if l != '.DS_Store']
        for file in dirs:
            img=np.array(tf_read_and_resize_image(data_path+"images/"+file,img_size))
            tf.keras.preprocessing.image.save_img(new_path+"/"+file,img)

def load_to_array(chexpert=False,img_size=32):
    """
    loads data from the initial csv's and images into 
    """
    if chexpert:
        data_path="/home/adam/Code/Datasets/chexpert/"### change to where you keep your data
        _DATA_DIR = "/home/adam/Code/Datasets/chexpert/CheXpert-v1.0-small" 
        _TRAIN_DIR = os.path.join(_DATA_DIR, "train")
        _VALIDATION_DIR = os.path.join(_DATA_DIR, "valid")
        _TRAIN_LABELS_FNAME = os.path.join(_DATA_DIR, "train.csv")
        _VALIDATION_LABELS_FNAME = os.path.join(_DATA_DIR, "valid.csv")

        filenames,labels=load_from_csv(data_path,_TRAIN_LABELS_FNAME,True)

        filenames_2,labels_2=load_from_csv(data_path,_VALIDATION_LABELS_FNAME,True)
        
        filenames.extend(filenames_2)
        arr=[]
        labels.extend(labels_2)

        arr= np.array([np.array(tf_load_image(img)) for img in filenames])
        np.save(data_path+"chexpert_"+str(img_size)+".npy",arr)
        np.save(data_path+"chexpert_"+str(img_size)+"_labels.npy",labels)
        return arr, np.array(labels)
    else:
        data_path="/home/adam/Code/Datasets/chestXray14/"### change to where you keep your data
        dirs = [l for l in os.listdir(data_path+"resized"+str(img_size)+"/") if l != '.DS_Store']
        arr=[]
        #arr= np.array([np.array(tf_load_image(data_path+"resized"+str(img_size)+"/"+img)) for img in dirs])
        y=make_xray14_labels()
        #np.save(data_path+"chestxray14_"+str(img_size)+".npy",arr)
        np.save(data_path+"chestxray14_"+str(img_size)+"_labels.npy",y)
        return arr, np.array(y)    
    
    
    
def load_from_csv(imgs_path,csv_path,resized=False):
    _LABELS = collections.OrderedDict({
            "-1.0": "uncertain",
            "1.0": "positive",
            "0.0": "negative",
            "": "unmentioned",
        })
    labeldict={"positive":1,"negative":0, "unmentioned":0,"uncertain":1} ### sets all uncertain labels to 1
    overlapping_labels=[0,2,5,6,8,10] ## according to pham et al. NF,CM,ED,CD,AC,PE
    ##### loads chexpert filenames and labels from files
    label_arr=[]
    arr=[]
    with tf.io.gfile.GFile(csv_path) as csv_f:
        reader = csv.DictReader(csv_f)
        # Get keys for each label from csv
        label_keys = reader.fieldnames[5:]

        for row in reader:
            # Get image based on indicated path in csv
            name = row["Path"]
            labels = [_LABELS[row[key]] for key in label_keys]
            labels_overlap=[labeldict[labels[i]] for i in overlapping_labels]
            if resized:
                A=name.split('/')[1:]
                path="resized32chex"
                for a in A:
                    path+="/"+a
                name=path
                   
            ## save the image_name and the label array
            label_arr.append(labels_overlap)
            
            arr.append(os.path.join(imgs_path, name))
        return arr,label_arr
def load_from_csv2(imgs_path,csv_path,resized=False):
    _LABELS = collections.OrderedDict({
            "-1.0": "uncertain",
            "1.0": "positive",
            "0.0": "negative",
            "": "unmentioned",
        })
    labeldict={"positive":1,"negative":0, "unmentioned":0,"uncertain":0} ### sets all uncertain labels to 0
    overlapping_labels=[0,2,5,6,8,10] ## according to pham et al. NF,CM,ED,CD,AC,PE
    ##### loads chexpert filenames and labels from files
    label_arr=[]
    arr=[]
    with tf.io.gfile.GFile(csv_path) as csv_f:
        reader = csv.DictReader(csv_f)
        # Get keys for each label from csv
        label_keys = reader.fieldnames[5:]

        for row in reader:
            # Get image based on indicated path in csv
            name = row["Path"]
            labels = [_LABELS[row[key]] for key in label_keys]
            labels_overlap=[labeldict[labels[i]] for i in overlapping_labels]
  
            ## save the image_name and the label array
            label_arr.append(labels_overlap)
            
            arr.append(os.path.join(imgs_path, name))
        return arr,label_arr

    
    
def make_chexpert_labels(data_path="/home/adam/Code/Datasets/chexpert/"):### change to where you keep your data

    def make_chex_onehot(row):
        _LABELS = collections.OrderedDict({
            "-1.0": 0, # uncertain
            "1.0": 1,  # positive
            "0.0": 0,  # negative
            "nan": 0,  # unmentioned
        }) ### sets all uncertain labels to 0, pass as policy parameter?
        label_keys=["No Finding","Cardiomegaly","Edema","Consolidation","Atelectasis","Pleural Effusion"]
        ## according to pham et al. NF,CM,ED,CD,AC,PE are the overlapping labels
        labels = [_LABELS[str(row[key])] for key in label_keys]
        if labels==[0,0,0,0,0,0]:
            labels=[1,0,0,0,0,0]
        return labels
    
    ##### loads chexpert filenames and labels from files
    # CheXpert-v1.0-small/
    
    sample=pd.read_csv(data_path+"new_chexpert/"+"train.csv")
    sample["Path"]=data_path+sample["Path"]
    sample["Finding_Labels"]=sample.apply(lambda row : make_chex_onehot(row), axis = 1)
    
    sample2=pd.read_csv(data_path+"new_chexpert/"+"valid.csv")
    sample2["Path"]=data_path+sample2["Path"]
    sample2["Finding_Labels"]=sample2.apply(lambda row : make_chex_onehot(row), axis = 1)
    ### merge train and validation samples and return
    sample=pd.concat([sample,sample2])
    
    
    return sample, np.array(sample["Finding_Labels"])
    
def make_xray14_labels(data_path="/home/adam/Code/Datasets/chestXray14/"):### change to where you keep your data
    """
    load the labels of chestXray14 and convert the labels to binary vectors which maps the occurrence of a label to a 1
    """
    
    data = pd.read_csv(data_path+"Data_Entry_2017_v2020.csv")
    
    sample = os.listdir(data_path+"images/")

    sample = pd.DataFrame({'Image Index': sample})

    sample = pd.merge(sample, data, how='left', on='Image Index')

    sample.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                      'Patient_Age', 'Patient_Gender', 'View_Position',
                      'Original_Image_Width', 'Original_Image_Height',
                      'Original_Image_Pixel_Spacing_X',
                      'Original_Image_Pixel_Spacing_Y']
    sample['Image_Index']=data_path+"images/"+sample['Image_Index']
    sample["Path"]= sample['Image_Index']
    
    def make_one_hot(label_string):
        labeldict={"No Finding":0,"Cardiomegaly":1,"Edema":2,"Consolidation":3,"Atelectasis":4,"Effusion":5}
        result=np.zeros(6)
        labels=label_string.split('|')
        for l in labels:
            if l not in ["No Finding","Cardiomegaly","Edema","Consolidation","Atelectasis","Effusion"]:
                pass
            else:
                result[labeldict[l]]=1
        return result.astype(int)

    sample['Finding_Labels'] = sample['Finding_Labels'].apply(lambda x: make_one_hot(x))
    y=sample['Finding_Labels']
    return sample, np.array(y)
