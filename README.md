# Tensor Pooling Driven Instance Segmentation Framework for Baggage Threat Recognition

## Introduction
This repository contains the implementation of our tensor pooling driven contour instance segmentation framework. 

![TP](/images/Picture212.png)

## Installation
To run the proposed framework, please download and install Anaconda. Afterward, please import the ‘environment.yml’ or alternatively install following packages: 
1. Python 3.7.4 
2. TensorFlow 2.2.0 (CUDA compatible GPU needed for GPU training) 
3. Keras 2.3.1 or above 
4. OpenCV 4.2 
5. Imgaug 0.2.9 or above 
6. Tqdm 

Both Linux and Windows OS are supported. To run some utility functions, please install MATLAB R2020a as well along with deep learning, image processing and computer vision toolbox.

## Datasets
The X-ray datasets can be downloaded from the following URLs: 
1. [GDXray](https://domingomery.ing.puc.cl/material/gdxray/) 
2. [SIXray](https://github.com/MeioJane/SIXray) 
3. [OPIXray](https://github.com/OPIXray-author/OPIXray) 

Each dataset contains the ground truths either in mat files, txt files or in xml files. To evaluate the proposed framework, the annotations must be in the mask form. To parse and convert the original box-level annotations to mask-level, we have provided the respective converters in the ‘…\utils’ folder. Please follow the same steps as mentioned below to prepare the training and testing data. 

## Dataset Preparation

1. Download the desired dataset. Resize the image through the 'resizer.m' script provided in the ‘…\utils’ folder.
2. Create the 'trainingDataset' and 'testingDataset' folders in the main code directory.
3. Create the 'trainingDataset\original' and 'testingDataset\original' folders, and put the resized training and testing images in these folders. 
4. Update the dataset paths in ‘tensorPooling.m’ file. These paths tell the location of the original images and where to save their multi-scale tensor representations.
5. Run the 'tensorPooling.m' file to generate the multi-scale tensors. These tensors will automatically be saved at the provided output path. 
5. Put the training annotations in '…\trainingDataset\train_annotations' folder (Note: Use parsers provided in the 'utils' folder to obtain these annotations). 
6. Take 10% portion of training images and put them in '…\trainingDataset\val_images' folder for validation purposes. The 10% threshold can be varied for other datasets/applications.
7. Put the respective annotations of validation images in '…\trainingDataset\val_annotations' folder 
9. Similarly, put test images in '…\testingDataset\test_images' folder and their annotations in '…\testingDataset\test_annotations' folder. 
4. The complete dataset hierarchy (structure) is given below:
```
├── trainingDataset
│   ├── original
│   │   └── or_image_1.png
│   │   └── or_image_2.png
│   │   ...
│   │   └── or_image_n.png
│   ├── train_images
│   │   └── tr_image_1.png
│   │   └── tr_image_2.png
│   │   ...
│   │   └── tr_image_n.png
│   ├── train_annotations
│   │   └── tr_image_1.png
│   │   └── tr_image_2.png
│   │   ...
│   │   └── tr_image_n.png
│   ├── val_images
│   │   └── va_image_1.png
│   │   └── va_image_2.png
│   │   ...
│   │   └── va_image_m.png
│   ├── val_annotations
│   │   └── va_image_1.png
│   │   └── va_image_2.png
│   │   ...
│   │   └── va_image_m.png
├── testingDataset
│   ├── original
│   │   └── or_image_1.png
│   │   └── or_image_2.png
│   │   ...
│   │   └── or_image_k.png
│   ├── test_images
│   │   └── te_image_1.png
│   │   └── te_image_2.png
│   │   ...
│   │   └── te_image_k.png
│   ├── test_annotations
│   │   └── te_image_1.png
│   │   └── te_image_2.png
│   │   ...
│   │   └── te_image_k.png
│   ├── segmentation_results
│   │   └── te_image_1.png
│   │   └── te_image_2.png
│   │   ...
│   │   └── te_image_k.png
```

## Training and Testing
1. Use '…\trainer.py' file to train the backbone network provided in the '…\codebase\models' folder. The training parameters can be configured in this file as well. Once the training is completed, the segmentation results are saved in the '…\testingDataset\segmentation_results' folder. These results are used by the 'instanceDetector.m' script in the next step for bounding box and mask generation. 
2. Once the step 1 is completed, please run '…\instanceDetector.m' to generate the final detection outputs. Please note that the '…\instanceDetector.m' requires that the original images are placed in the '…\testingDataset\original' folder (as discussed in the previous section).

## Results
The additional results of the proposed framework are presented in the '…\results' folder. Please feel free to email us if you require the trained instances. 

## Citation
If you use the proposed framework (or any part of this code in your research), please cite the following paper:

```
@inproceedings{tensorPooling,
  title   = {Tensor Pooling Driven Instance Segmentation Framework for Baggage Threat Recognition},
  author  = {Taimur Hassan and Samet Akcay and Mohammed Bennamoun and Salman Khan and Naoufel Werghi},
  note = {Submitted in Springer Neural Computing and Applications},
  year = {2021}
}
```

## Contact
If you have any query, please feel free to contact us at: taimur.hassan@ku.ac.ae.
