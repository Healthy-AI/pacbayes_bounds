import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.image import resize
from PIL import Image
from data import tasks
import time
source_image_dir=os.getcwd()+"../Datasets/"# where the datasets are stored
class CheXpertDataGenerator(Sequence):
    'Data Generator for CheXpert and chestXray14'
    
    def __init__(self, dataset_df, y, iw, batch_size=32,
                 target_size=(224, 224),  verbose=0,
                 shuffle_on_epoch_end=False, random_state=1):
        self.dataset_df = dataset_df
        self.y=y
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.x_path=dataset_df["Path"]
        self.iw=np.array(iw)
        self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
    def __bool__(self):
        return True

    def __len__(self):
        return self.steps
    def __getitem__(self, idx):
        
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path]).astype(np.float32)
        
        batch_x = self.transform_batch_images(batch_x)        
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        img = tf.io.read_file(image_path)

        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        image_array = tf.image.convert_image_dtype(img, tf.float32)
        image_array = tf.image.resize(image_array, self.target_size)
        return image_array

    def transform_batch_images(self, batch_x):
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        if batch_x.shape == imagenet_mean.shape:
            batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
