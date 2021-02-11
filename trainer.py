import cv2
import os
from keras.models import load_model
import matplotlib.pyplot as plt

from codebase.models.net import *
from codebase.models.segnet import *
from codebase.models.pspnet import *
from codebase.models.unet import *
from codebase.models.fcn import *

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

model = net(n_classes=20 ,  height=576, width=768)

#model = resnet50_segnet(n_classes=20 ,  input_height=576, input_width=768)

model.train(
    train_images =  "trainingDataset/train_images/",
    train_annotations = "trainingDataset/train_annotations/",
	val_images =  "trainingDataset/val_images/",
    val_annotations = "trainingDataset/val_annotations/",
    checkpoints_path = None , epochs=50, validate=True
)

folder = "testingDataset/test_images/"
for filename in os.listdir(folder):
	out = model.predict_segmentation(inp=os.path.join(folder,filename),
	out_fname=os.path.join("testingDataset/segmentation_results/",filename))

print(model.evaluate_segmentation( inp_images_dir="testingDataset/test_images/"  , 
	annotations_dir="testingDataset/test_annotations/" ) )