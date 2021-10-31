---
title: "Deep learning CPU vs GPU "
teaching: 40
exercises: 10
questions:
- "How does the performance of GPU compare with that of CPU?"
- "How to use Mana to do Machine Learning research?"
- "How to ask for computing resources?"
objectives:
- "Do a basic Deep Learning tutorial on Mana"
keypoints:
- "XXXXXXXXXX" 
---

## Jupyter Lab as an Interactive Application in Open OnDemand
As we previously saw, Open OnDemand allows us to use interactive applications, one of which is Juypter Lab.
  {% include figure.html url="" max-width="50%"
   file="/fig/ood_form.png"
   alt="Connect to cluster" caption="" %}
The form is used to specify what resources you want, which are then placed into a queue with other waiting jobs and will start to run your job 
as soon as the resources requested are available.  
> ## Under the hood
> 
> The Open On Demand form for interactive applications define a job script and passes it to the HPC systems job scheduler, leaving the
> hard work of how to start the application on the HPC system and how to write a job script that the job scheduler can understand.
>
{: .callout}
                                             
> ## Starting an Interactive session of Jupyter Lab
>
> As we will be working in Jupyter Lab to explore some concepts when working with HPC systems and deep learning your challenge is to 
> start an interactive application of Jupyter Lab with the following parameters
> * Partition: workshop
> * Number of hours: 5
> * Number of cores: 6
> * GB of Ram: 20 GB
> * Number of GPUs requested: 1
> * GPU Type: Any
>
{: .challenge}                                             
                                             
  
## Jupyter Lab vs Jupyter Notebook

Jupyter notebook allows you to access .ipynb files only, i.e. it will create a computational environment which stores your code, results, plots, texts etc. And here you can work only in one of your environment. But Jupyter Lab gives a better user interface along with all the facilties provided by the notebook. It has a modular structure where one can access .py, .ipynb, html or markdown files, access filebrowser, all in the same window. 
  
### Jupyter Lab
It is a flexible, web based application which is mainly used in data science and machine learning research. It gives you acess to file browser (to upload, download, copy, rename, delete files), do data visualization, add data, code, texts, equations all in one place, use big data tools, share your work with others. It supports more than 40 programming languages and has an interactive output. 
  
> **Q. How does it work?**
>  
>You write your code or plain text in rectangular “cells” and the browser then passes it to the back-end “kernel”, which runs your code and returns output.

`Note: .ipynb file is a python notebook which stores code, text, markdown, plots, results in a specific format but .py file is a python file which only 
stores code and plain text (like comments etc).`  


  
## How to access and install softwares and modules on cluster?
  
### Package Managers:
Software packages already installed on the cluster which we can use to install required libraries, softwares and can even choose which version to install.
You can use following commands to see what modules are available on the cluster or which ones are already loaded or to load a specific module in your environment:

~~~
  module avail
  module list 
  module load <MODULE_NAME>
~~~
{: .source}
  
1. Pip: tool to install Python software packages only. 

2. Anaconda (or Conda): cross platform package and environment manager which lets you access C, C++ libraries, R package for scientific computing along with Python.
  
`Note: package contains all the files you need for a module`  

### Anaconda
- allows you to install softwares written in any programming language,
- flexibility to create different environments with different software versions,
- can use both CLI and GUI
  
> If you try to access a library with different version based on your project, pip may throw an error. To create isolated environments you can use virtual environment (venv) with pip.
  
> ## Environment setup
>  
> 1. Create a conda environment
> ~~~
> * module load lang/Anaconda3
> * conda create --name tf2
> * source activate tf2
> ~~~
> 2. Download libraries
> ~~~
> * conda install tensorflow-gpu
> * conda install matplotlib
> * conda install tensorflow
> * conda install keras
> ~~~
> 3. Get a python kernel
> ~~~
> * conda install ipykernel
> * python -m ipykernel install --user --name tf2 --display-name tf2
> ~~~
> 
{: .challenge} 
  
## Deep Learning Tutorial

This is a basic image classification tutorial from CIFAR-10 dataset using tensorflow. 
  
### Q. What is CIFAR-10 dataset?
  
CIFAR-10 is a common dataset used for machine learning and computer vision research. It is a subset of 80 million tiny image dataset and consists of 60,000 images. The images are labelled with 10 different classes. So each class has 5000 training images and 1000 test images. Each row represents a color image of 32 x 32 pixels with 3 channels (RGB).   
  
 <img src="../fig/CIFAR-10.jpg" width="500" height="400">
 
> **What is Tensorflow?** 
>
> It is an open source software used in machine learning particularly for training neural networks. We'll define our model using 'Keras'- a high level API which acts as an interface between tensorflow and python and makes it easy to build and train models.
{: .callout}
  
### Basic workflow of Machine Learning
  
1. Collect the data
2. Pre-process the data
3. Define a model
4. Train the model
5. Evaluate/test the model
6. Improve your model

> ### How to check if you're using GPU ?
> ~~~
> tf.config.list_physical_devices('GPU')
> ~~~
> Now, how would you check for CPU ?
>
> > ### Solution
> > ~~~
> > tf.config.list_physical_devices('CPU')
> > ~~~
> {: .solution}
{: .challenge}

> ## Working with Cifar-10 dataset
>
> * Import all the relevant libraries
> 
> ~~~
> import numpy as np
> import matplotlib.pyplot as plt
> 
> import tensorflow as tf
> import h5py
> import keras
> from keras.datasets import cifar10
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, InputLayer, Dropout
> import keras.layers.merge as merge
> from keras.layers.merge import Concatenate
> from tensorflow.keras.utils import to_categorical
> from tensorflow.keras.optimizers import SGD, Adam
> 
> %matplotlib inline
> ~~~
> 
> * Check for CPU and GPU
> 
> * Load the data and analyze its shape
> 
> ~~~
> (x_train, y_train), (x_valid, y_valid) = cifar10.load_data()
> nb_classes = 10
> class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
> print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
> print('Test: X=%s, y=%s' % (x_valid.shape, y_valid.shape))
> print('number of classes= %s' %len(set(y_train.flatten())))
> print(type(x_train))
> ~~~
> 
> * Plot some examples 
> ~~~
> plt.figure(figsize=(8, 8)) 
> for i in range(2*7):
>     # define subplot
>     plt.subplot(2, 7, i+1)
>     plt.imshow(x_train [i])
>     class_index = np.argmax(to_categorical(y_train[i], 10))
>     plt.title(class_names[class_index], fontsize=9)
> ~~~    
> * Convert data to HDF5 format
> ~~~
> with h5py.File('dataset_cifar10.hdf5', 'w') as hf:
>     dset_x_train = hf.create_dataset('x_train', data=x_train, shape=(50000, 32, 32, 3), compression='gzip', chunks=True)
>     dset_y_train = hf.create_dataset('y_train', data=y_train, shape=(50000, 1), compression='gzip', chunks=True)
>     dset_x_test = hf.create_dataset('x_valid', data=x_valid, shape=(10000, 32, 32, 3), compression='gzip', chunks=True)
>     dset_y_test = hf.create_dataset('y_valid', data=y_valid, shape=(10000, 1), compression='gzip', chunks=True)
> ~~~
> * Define the model
> 


{% include links.md %}
