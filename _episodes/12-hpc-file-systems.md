---
title: "Staging and File System Choice"
teaching: 15
exercises: 25
questions:
- "What is a file system?"
- "What is a distributed file system?"
- "How do you optimize file system on MANA?"
objectives:
- "Understand general file system and distributed file system concepts."
- "Being able to stage files on MANA lustre scratch."
keypoints:
- "File system is a way in which operating systems organize data on storage devices."
- "Distributed file system organizes data accross network attached storage devices."
- "Distributed file system has the advantage of supporting larger scale storage capacity."
- "MANA supports lustre and NFS file systems."
- "Lustre on MANA is setup with solid state drives."
- "NFS on MANA is setup with spinning drives."
---

## What is a file system?

The term file system is ubiquitous among operating system users. At some point in time, you may have had to format a USB flash drive and may have to choose between different file systems such as FAT32 or NTFS (on a Windows OS), exFat or Mac OS Extended (on a Mac). 

&nbsp;

{% include figure.html url="" max-width="40%"
   file="/fig/inode.png"
   alt="Inode" caption="Inode Table (https://www.bluematador.com/blog/what-is-an-inode-and-what-are-they-used-for)
 "%}

&nbsp;

File systems are ways in which operating systems organize data (i.e. documents, movies, music) on storage devices. From the storage device hardware perspective, data is represented as a sequence of blocks of bytes, however that by itself isn’t useful to regular users. Regular users mostly care about file and directory level abstraction and rarely work with blocks of bytes! Luckily, the file system under the hood handles the organization and locating logic of these blocks of bytes for the users so that you can run your favorite “ls” command. For example in a Linux file system, “file” information is stored in an inode table (illustrated in figure above) as sort of a lookup table for locating corresponding blocks of a particular file on disk. Located blocks are combined together to form a single file which the end user can work with. 


## What is a distributed file system?

On a cluster, blocks of data that make up a single file are distributed among network attached storage or NAS devices. Similar principles apply here, the goal of a cluster file system is still to organize and locate blocks of data (but across network!) and present them to the end user as one contiguous file. One main benefit of stringing storage devices together into a network connected cluster is to increase storage capacity beyond what a single computer can have. Imagine working with 100 TB files on your laptop. Of course, the storage can be shared with different cluster users further increasing utilization of these storage devices. 

On Mana cluster, the two main supported file systems are <b>Network File System (NFS)</b> and <b>Lustre file</b> systems.To us MANA users, we have 2 special folder called lustre_scratch and nfs_scratch where we can temporarily store our data on the cluster. Note that the scratch folders will be purged after some period of time, please save your important files into your home directory.

> ## Locate Lustre and NFS File System Scratch On MANA
> Let's locate our lustre and nfs scratch folder.
> First create a new cell in our Jupyter Lab Notebook after the first cell
> Paste the code into the cell and run.
> The pwd command will print where you are in the directory tree and it should be something like "/home/username
> The ls command will print files and directory that are available in your home/username directory. You should see "lus_scratch" and "nfs_scratch" listed in the print out.
> Note that the '!' symbol below allows us to execute terminal commands in our Jupyter Lab Notebook cell.
> ~~~
> {% raw  %}
> !pwd
> !ls
> {% endraw %}
> ~~~
{: .challenge}

### Lustre File System.
&nbsp;

{% include figure.html url="" max-width="40%"
   file="/fig/lustre.png"
   alt="Lustre File System" caption="Lustre File System (https://wiki.lustre.org/Introduction_to_Lustre)
 "%}

&nbsp;

Lustre is a parallel distributed file system, where file operations are distributed across multiple file system servers. In Lustre, storage is divided among multiple servers allowing for ease of scalability and fault tolerance. This file system is great for random access performance.

### NFS File System.

&nbsp;

{% include figure.html url="" max-width="40%"
   file="/fig/nfs.png"
   alt="NFS File System" caption="NFS File System (https://www.geeksforgeeks.org/what-is-dfsdistributed-file-system/)
 "%}

&nbsp;

NFS is a single server distributed file system where file operations are not parallel across servers but a single server serves requests to the cluster. NSF is an older technology and has the advantage of having gone through the test of time and is trusted among cluster architects. 

## Choosing the Right File System For Performance.

Depending on the user's need, different file systems are optimized for different purposes. One may be optimized for random access speed, one with error correcting capability, or one with high redundancy to prevent loss of data. 
For this workshop, we will focus on disk random access speed MANA cluster.

On MANA we have 3 locations for file storage: "home/user", "lus_scratch", and "nfs_scratch" folders available to us. 
On MANA, our "home/user" directory resides on an NFS file system server. Though "home/user" is great for storing our files on the cluster, it's serverly lacking in read/write performance. For the best performance (read/write speed) we will mostly want to use "lus_scratch". As discussed above, the reason is because Lustre file system distributes workload accross multiple meta servers, while NFS is a single server file system. In addition, MANA Lustre file system is configured with solid state drives while NFS file system is configured with hardrives, improving MANA lustre file system read/write speed further.

> ## List Usage Information
> On MANA, there's a command that lists disk usage on the cluster.
> Create a new cell below our previous block.
> Paste the code into the cell and run.
> You should see a table that lists disk space used and remaining space.
> ~~~
> {% raw  %}
> !usage
> {% endraw %}
> ~~~
{: .challenge}

> ## Solid State Drive vs. Hardrive Performance
> Solid state drives can read up to 10 times faster and writes up to 20 times faster than hard disk drive. 
> You can read more about it [here](https://www.avg.com/en/signal/ssd-hdd-which-is-best#:~:text=A%20solid%20state%20drive%20reads,PCIe%203.0%20to%204.0%20connectors).
{: .callout}  

## Optimizing Our Deep Learning Code
Since our Jupyter Lab Notebook is somewhat IO bounded (reading/writing images to disks) we can optimize it further by utilizing "lus_scratch" folder. Instead of reading and writing training data from our home directory, we're going to instead work from the "lus_scratch" directory. 

> ## Stage Training Files
> We will re-download the training data into the "lus_scratch" directory. 
> Create a new cell in our Jupyter Lab Notebook.
> Paste the code below into the cell.
> Update the path so that the data is save into our "lus_scratch" directory.
> You should see a print out of the with "dataset_cifar10.hdf5".
> ~~~
> {% raw  %}
> # Stage files onto luster directory.
> (x_train_lus, y_train_lus), (x_valid_lus, y_valid_lus) = cifar10.load_data()
> with h5py.File('./lus_scratch/dataset_cifar10.hdf5', 'w') as hf:
>   hf.create_dataset('x_train', data=x_train_lus, shape=(50000, 32, 32, 3), compression='gzip', chunks=True)
>   hf.create_dataset('y_train', data=y_train_lus, shape=(50000, 1), compression='gzip', chunks=True)
>   hf.create_dataset('x_valid', data=x_valid_lus, shape=(10000, 32, 32, 3), compression='gzip', chunks=True)
>   hf.create_dataset('y_valid', data=y_valid_lus, shape=(10000, 1), compression='gzip', chunks=True)
> !ls "./lus_scratch/"
> {% endraw %}
> ~~~
{: .challenge}

> ## Update Training Data Path 
> Now that we have successfully staged our training data to the "lus_scratch" directory, let us now update the data path for our model.
> Find where the data is loaded, the code in the cell should look like the code below.
> Once the cell is updated, run the cell.
> ~~~
> {% raw  %}
> # Loading training data from luster scratch
>with h5py.File('./lus_scratch/dataset_cifar10.hdf5', 'r') as hf:
>    X_train = hf['x_train'][:]
>    Y_train = hf['y_train'][:]
>    X_valid = hf['x_valid'][:]
>    Y_valid = hf['y_valid'][:]
>print(type(X_train))
> {% endraw %}
> ~~~
{: .challenge}


> ## Duplicate the Training Cell
> Before retraining our model, let's duplicate the cell that contain codes for training our model.
> This is so that we don't lose the printout from our previous run. Use the GPU version of the training code similiar to below.
> Run training step of the model.
> Compare the training time.
> ~~~
> {% raw  %}
> # Running on Lustre 
> with tf.device('/device:GPU:0'):
>    history = model.fit(data_train,epochs=20,
>                        verbose=1, validation_data=data_valid)
> {% endraw %}
> ~~~
{: .challenge}


{% include links.md %}
