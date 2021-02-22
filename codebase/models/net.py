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
from .net2 import get_segmentor2 
from .model_utils import get_segmentation_model, resize_image
    
def net(n_classes,  height=384, width=576, addActivation = True):

    modelSegmentor = get_segmentor2(n_classes, height=height, width=width, addActivation = addActivation)
    modelSegmentor.model_name = "segmentor"
	
    return modelSegmentor

def get_segmentor(classes, height=384, width=576, addActivation = True):

    assert height % 192 == 0
    assert width % 192 == 0

    inputLayer, features = encoder(height=height,  width=width)
    
    o = decoder(features, classes)
	
    modelSegmentor = get_segmentation_model(inputLayer, o)
    return modelSegmentor
    
def SPB(x, inputLayer, numFilt, kernel, stride, bn_axis, order, rescaleInput = False):
    x_i = x
    #print(['########', str(x), '########'])
    x = Conv2D(numFilt, (kernel, kernel), data_format=order, strides=(stride, stride), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(numFilt, (kernel, kernel), data_format=order, strides=(stride, stride), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(4*numFilt, (kernel, kernel), data_format=order, strides=(stride, stride), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    #print(['########', str(x), '########'])
    
    x_i = Conv2D(4*numFilt, (2*kernel, 2*kernel), data_format=order, strides=(stride, stride), padding='same')(x_i)
    x_i = BatchNormalization(axis=bn_axis)(x_i)
    
    x = layers.add([x, x_i])
    
    x_2 = Conv2D(4*numFilt, (2*kernel, 2*kernel), data_format=order, strides=(2*stride, 2*stride), padding='same')(inputLayer)
    x_2 = BatchNormalization(axis=bn_axis)(x_2)
    
    if rescaleInput == True:
        x_2 = MaxPooling2D((10, 10), data_format=order, strides=(4, 4))(x_2)
    else:
        x_2 = MaxPooling2D((10, 10), data_format=order, strides=(2, 2))(x_2)
    
    x = layers.multiply([x, x_2])
    
    x = Activation('relu')(x)
    
    return x
    
    
def IB(x, numFilt, kernel, stride, bn_axis, order):
    x_i = x
    #print(['########', str(x), '########'])
    x = Conv2D(numFilt, (kernel, kernel), data_format=order, strides=(stride, stride), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(numFilt, (kernel, kernel), data_format=order, strides=(stride, stride), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(4*numFilt, (kernel, kernel), data_format=order, strides=(stride, stride), padding='same')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    #print(['########', str(x), '########'])
    
    x = layers.add([x, x_i])
    
    return x
        
    
def encoder(height,  width):
    from .config import IMAGE_ORDERING as order
    
    assert height > 0 == 0
    assert width > 0 == 0

    if order == 'channels_first':
        inputLayer = Input(shape=(3, height, width))
    elif order == 'channels_last':
        inputLayer = Input(shape=(height, width, 3))

    if order == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    kernel_size = 3
    
    x = Conv2D(64, (9, 9), data_format=order, strides=(2, 2))(inputLayer)
    #print(['########', str(x), '########'])
    
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    s1 = x
    print(['########', str(s1), '########'])
    
    x = MaxPooling2D((6, 6), data_format=order, strides=(2, 2))(x)
    x=SPB(x, inputLayer, 64, 1, 1, bn_axis, order)
    #print(['########', str(x), '########'])
    x=IB(x, 64, 1, 1, bn_axis, order)
    #print(['########', str(x), '########'])
    x = Conv2D(256, (3, 3), data_format=order, strides=(2, 2), padding='same')(x)
    s2 = x
    print(['########', str(s2), '########'])
    
    x=SPB(x, inputLayer, 64, 1, 1, bn_axis, order, rescaleInput = True)
    x=IB(x, 64, 1, 1, bn_axis, order)
    s3 = x
    print(['########', str(s3), '########'])
    
    return inputLayer, [s1, s2, s3]
    
def decoder(features, classes):
    from .config import IMAGE_ORDERING as order

    [first, second, lastFeatures] = features
    
    print(lastFeatures)
    x = AveragePooling2D((2, 2), data_format=order, strides=(1, 1))(lastFeatures)
    x = Conv2D(256, (1, 1), data_format=order, strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = resize_image(x, (2, 2), data_format=order)
    x = layers.add([x, second])
    print(x)
    
    x = Conv2D(64, (1, 1), data_format=order, use_bias=False)(x)
    x = BatchNormalization()(x)
    print(x)
    x = resize_image(x, (2, 2), data_format=order)
    x = ZeroPadding2D((2, 2), data_format=order)(x)
    x = resize_image(x, (2, 2), data_format=order)
    x = Conv2D(64, (5, 5), data_format=order, use_bias=False)(x)
    x = layers.add([x, first])
    x = Conv2D(512, (1, 1), data_format=order, use_bias=False)(x)
    x = BatchNormalization()(x)
    #x = Activation('softmax')(x) # softmax is added during model assembly in 'get_segmentation_model()'
        
    return x