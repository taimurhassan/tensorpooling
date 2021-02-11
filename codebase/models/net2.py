# Proposed model
import numpy as np
import keras
from keras.models import *
from keras.models import Model
from keras.layers import *
import keras.backend as K
 
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import tensorflow as tf
from .resnet50 import get_resnet50_encoder 
from .model_utils import get_segmentation_model, resize_image

def net(n_classes,  height=384, width=576, addActivation = True):

    modelSegmentor = get_segmentor2(n_classes, height=height, width=width, addActivation = addActivation)
    modelSegmentor.model_name = "segmentor"
	
    return modelSegmentor
    
def get_segmentor2(classes, height=384, width=576, addActivation = True):

    assert height % 192 == 0
    assert width % 192 == 0

    inputLayer, levels = get_resnet50_encoder(input_height=height,  input_width=width)
    [f1, f2, f3, f4, features] = levels
    
    o = decoder(features, classes)
    
    modelSegmentor = get_segmentation_model(inputLayer, o)
    return modelSegmentor
    
def decoder(features, classes):
    from .config import IMAGE_ORDERING as order

    pool_factors = [1, 2, 8, 16]
    list = [features]
    print(len(list))
    
    if order == 'channels_first':
        h = K.int_shape(features)[2]
        w = K.int_shape(features)[3]
    elif order == 'channels_last':
        h = K.int_shape(features)[1]
        w = K.int_shape(features)[2]

    pool_size = strides = [
    int(np.round((float(h)+ 1)/3)),
    int(np.round(((float(w) + 1)/3)))]

    pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(features)
    pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
    pooledResult = BatchNormalization()(pooledResult)
    pooledResult = Activation('relu')(pooledResult)
        
    pooledResult = resize_image(pooledResult, strides, data_format=order)
    #list[0] = pooledResult
        
    for p in pool_factors:
        if order == 'channels_first':
            h = K.int_shape(features)[2]
            w = K.int_shape(features)[3]
        elif order == 'channels_last':
            h = K.int_shape(features)[1]
            w = K.int_shape(features)[2]

        pool_size = strides = [
            int(np.round((float(h))/ p)),
            int(np.round((float(w))/ p))]

        pooledResult = AveragePooling2D(pool_size, data_format=order,
                         strides=strides, padding='same')(features)
        pooledResult = Conv2D(512, (1, 1), data_format=order,
               padding='same', use_bias=False)(pooledResult)
        pooledResult = BatchNormalization()(pooledResult)
        pooledResult = Activation('relu')(pooledResult)
        
        pooledResult = resize_image(pooledResult, strides, data_format=order)
        list.append(pooledResult)
		
    if order == 'channels_first':
        features = Concatenate(axis=1)(list)
    elif order == 'channels_last':
        features = Concatenate(axis=-1)(list)

    features = Conv2D(512, (1, 1), data_format=order, use_bias=False)(features)
    features = BatchNormalization()(features)
    features = Activation('relu')(features)

    features = Conv2D(classes, (3, 3), data_format=order,
               padding='same')(features)
    features = resize_image(features, (8, 8), data_format=order)

    return features