"""
This code generates one weight file stored in the current dir (./weights.h5). 

"""

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import cv2
import sys

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
os.environ["CUDA_VISIBLE_DEVICES"]="3"

size= 256
batch_size=16

#ss = 10

m1= {}
sd1 = {}
m2 = {}
sd2 = {}

MEAN_SD_ref=open('/mnt/ibrixfs01-MRI/analysis/washen/temp/mean_sd.txt','r')
for line in MEAN_SD_ref:
    line=line.rstrip()
    table=line.split('\t')
    key=table[0]
    m1[key]=float(table[1])
    sd1[key]=float(table[2])
    m2[key]=float(table[3])
    sd2[key]=float(table[4])

MEAN_SD_ref.close()

def preprocess(imgs):
    #imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    #imgs_p = np.ndarray((imgs.shape[0], size), dtype=np.uint8)
    imgs_p=imgs
    #for i in range(imgs.shape[0]):
       # imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
    #    imgs_p[i] = resize(imgs[i], size, preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

import unet
import random
model = unet.get_unet()
import GS_split

train_x_path = '/mnt/ibrixfs01-MRI/analysis/washen/temp/train/T1T2/'
train_y_path = '/mnt/ibrixfs01-MRI/analysis/washen/temp/train/Segmentation/'

def generate_data(train_line, batch_size):
    """Replaces Keras' native ImageDataGenerator."""
    i = 0
    while True:
        image_batch = []
        label_batch = []
    
        for b in range(batch_size):
            if i == len(train_line):
                i = 0
                random.shuffle(train_line)
                
            sample = train_line[i]
            i += 1
            
            image = np.zeros((size,size,2))           
            label = np.zeros((size,size,1))
            
            imgName = sample.replace('seg','img')

            img = np.load(train_x_path+imgName)
            key = sample.split('-seg')[0]
            
            mmm=m1[key]
            sss=sd1[key]
            image[:,:,0]=(img[0,:,:]-mmm)/sss

            mmm=m2[key]
            sss=sd2[key]
            image[:,:,1]=(img[1,:,:]-mmm)/sss

            rrr_flipup=random.random()
            if (rrr_flipup>0.5):
                image=np.flipud(image)

            rrr_fliplr=random.random()
            if (rrr_fliplr>0.5):
                image=np.fliplr(image)

            image_batch.append(image)


            if ('neg' in sample):
                pass
            else:
                label[:,:,0]=np.load(train_y_path+sample)                

            if (rrr_flipup>0.5):
                label=np.flipud(label)

            if (rrr_fliplr>0.5):
                label=np.fliplr(label)

            label_batch.append(label)
            
        label_batch=np.array(label_batch)
        image_batch=np.array(image_batch)
        yield (image_batch, label_batch)

#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=False)
seed = [556,40,23,95,3] #seed for splitting training and testing set
for i in range(1):

    (train_line,test_line)=GS_split.GS_split('/mnt/ibrixfs01-MRI/analysis/washen/temp/train/Segmentation/',seed[i])


    callbacks = [
        keras.callbacks.TensorBoard(log_dir='./log/121318',
        histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(os.path.join('./weights/121318', 'weights_%d.h5'%i),
        verbose=0, save_weights_only=True)#,monitor='val_loss')
        ]

    model.fit_generator(
        generate_data(train_line, batch_size),
        #steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=20,validation_data=generate_data(test_line,batch_size),validation_steps=100,callbacks=callbacks)
        steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=10,validation_data=generate_data(test_line,batch_size),validation_steps=100,callbacks=callbacks)

