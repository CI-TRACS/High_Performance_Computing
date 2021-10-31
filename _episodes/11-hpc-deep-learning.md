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
> 
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
> 
> ~~~
> plt.figure(figsize=(8, 8)) 
> for i in range(2*7):
>     # define subplot
>     plt.subplot(2, 7, i+1)
>     plt.imshow(x_train [i])
>     class_index = np.argmax(to_categorical(y_train[i], 10))
>     plt.title(class_names[class_index], fontsize=9)
> ~~~    
> 
> * Convert data to HDF5 format
> 
> ~~~
> with h5py.File('dataset_cifar10.hdf5', 'w') as hf:
>     dset_x_train = hf.create_dataset('x_train', data=x_train, shape=(50000, 32, 32, 3), compression='gzip', chunks=True)
>     dset_y_train = hf.create_dataset('y_train', data=y_train, shape=(50000, 1), compression='gzip', chunks=True)
>     dset_x_test = hf.create_dataset('x_valid', data=x_valid, shape=(10000, 32, 32, 3), compression='gzip', chunks=True)
>     dset_y_test = hf.create_dataset('y_valid', data=y_valid, shape=(10000, 1), compression='gzip', chunks=True)
> ~~~
> 
> * Define the model
> 
> ~~~
> model = tf.keras.Sequential()
> model.add(InputLayer(input_shape=[32, 32, 3]))
>
> model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
> model.add(MaxPooling2D(pool_size=[2,2], strides=[2, 2], padding='same'))
>
> model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
> model.add(MaxPooling2D(pool_size=[2,2], strides=[2, 2], padding='same'))
>
> model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
> model.add(MaxPooling2D(pool_size=[2,2], strides=[2, 2], padding='same'))
>
> model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
> model.add(MaxPooling2D(pool_size=[2,2], strides=[2, 2], padding='same'))
>
> model.add(Flatten())
>
> model.add(Dense(256, activation='relu'))
> model.add(Dropout(0.2))
>
> model.add(Dense(512, activation='relu'))
> model.add(Dropout(0.2))
>
> model.add(Dense(10, activation='softmax'))
>
> model.summary()
> ~~~
>
> * Define the data generator
>
> ~~~
> class DataGenerator(tf.keras.utils.Sequence):
>    
>     def __init__(self, batch_size, test=False, shuffle=True):
>        
>         PATH_TO_FILE = 'dataset_cifar10.hdf5'
>        
>         self.hf = h5py.File(PATH_TO_FILE, 'r')         
>         self.batch_size = batch_size
>         self.test = test
>         self.shuffle = shuffle
>        self.on_epoch_end()
>
>     def __del__(self):
>         self.hf.close()
>        
>     def __len__(self):
>         return int(np.ceil(len(self.indices) / self.batch_size))
>
>     def __getitem__(self, idx):
>         start = self.batch_size * idx
>         stop = self.batch_size * (idx+1)
>        
>         if self.test:
>             x = self.hf['x_valid'][start:stop, ...]
>             batch_x = np.array(x).astype('float32') / 255.0
>             y = self.hf['y_valid'][start:stop]
>             batch_y = to_categorical(np.array(y), 10)
>         else:
>             x = self.hf['x_train'][start:stop, ...]
>             batch_x = np.array(x).astype('float32') / 255.0
>             y = self.hf['y_train'][start:stop]
>             batch_y = to_categorical(np.array(y), 10)
>
>         return batch_x, batch_y
>
>     def on_epoch_end(self):
>         if self.test:
>             self.indices = np.arange(self.hf['x_valid'][:].shape[0])
>         else:
>             self.indices = np.arange(self.hf['x_train'][:].shape[0])
>            
>         if self.shuffle:
>             np.random.shuffle(self.indices)
>  ~~~
>
> Generate batches of data for training and validation dataset
>
> ~~~
> batchsize  = 250 
> data_train = DataGenerator(batch_size=batchsize)
> data_valid = DataGenerator(batch_size=batchsize, test=True, shuffle=False)
> ~~~
> 
> First, let's train the model using CPU
> ~~~
> with tf.device('/device:CPU:0'):
>     history = model.fit(data_train,epochs=10,
>                         verbose=1, validation_data=data_valid)
> ~~~
>                         
> Now, lets try with GPU to compare its performance with CPU
>
> ~~~
> from tensorflow.keras.models import clone_model
> new_model = clone_model(model)
> opt = keras.optimizers.Adam(learning_rate=0.001)
> new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])  
>                         
> % train the new model with GPU
> ~~~
> 
> > ### Solution
> > ~~~
> > with tf.device('/device:GPU:0'):
> >     new_history = new_model.fit(data_train,epochs=10,
> >                                 verbose=1, validation_data=data_valid)
> >  ~~~                               
> {: .solution}
>                                  
> Plotting the losses and accuracy for training and validation set
>
> ~~~
> fig, axes = plt.subplots(1,2, figsize=[16, 6])
> axes[0].plot(history.history['loss'], label='train_loss')
> axes[0].plot(history.history['val_loss'], label='val_loss')
> axes[0].set_title('Loss')
> axes[0].legend()
> axes[0].grid()
> axes[1].plot(history.history['accuracy'], label='train_acc')
> axes[1].plot(history.history['val_accuracy'], label='val_acc')
> axes[1].set_title('Accuracy')
> axes[1].legend()
> axes[1].grid()
> ~~~
>
> Evaluate the model and make predictions
> 
> ~~~
> x = x_valid.astype('float32') / 255.0
> y = to_categorical(y_valid, 10)
> score = new_model.evaluate(x, y, verbose=0)
> print('Test cross-entropy loss: %0.5f' % score[0])
> print('Test accuracy: %0.2f' % score[1])
> 
> y_pred = new_model.predict_classes(x)
> ~~~
> 
> Plot the predictions
>
> ~~~
> plt.figure(figsize=(8, 8)) 
> for i in range(20):
>     plt.subplot(4, 5, i+1)
>     plt.imshow(x[i].reshape(32,32,3))
>     index1 = np.argmax(y[i])
>     plt.title("y: %s\np: %s" % (class_names[index1], class_names[y_pred[i]]), fontsize=9, loc='left')
>     plt.subplots_adjust(wspace=0.5, hspace=0.4)
>  ~~~   



{% include links.md %}
