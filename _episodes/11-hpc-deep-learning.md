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

## Connecting to UH-HPC cluster Mana (SSH vs OOD)

On logging into a cluster, you are into a login node which has limited storage and restricted use because it is being shared by everybody so you submit your jobs to a compute node to get required resources for your research work. These compute nodes have lots of storage. There are 2 ways to access the cluster:
1. Using Secure Shell (SSH)
   - done using Linux like commands in terminal (alo known as Command Line Interface or CLI),
   - login using **ssh <username>@mana.its.hawaii.edu**,
   - works with weak connections too.
  
  <img src="fig/CLI.png" width="300" height="400">
  
2. Open On Demand (OOD)
   - a web based interface (or graphical user interface (GUI),
   - easy to access,
   - use interactive apps like Jupyter, RStudio and even Linux shell,
   - easy access to file systems as well.
  
 <img src="fig/MANA.png" width="300" height="400">
                                             
  
## Jupyter Lab vs Jupyter Notebook

Jupyter notebook allows you to access .ipynb files only, i.e. it will create a computational environment which stores your code, results, plots, texts etc. And here you can work only in one of your environment. But Jupyter Lab gives a better user interface along with all the facilties provided by the notebook. It has a modular structure where one can access .py, .ipynb, html or markdown files, access filebrowser, all in the same window. 
  
### Jupyter Lab
It is a flexible, web based application which is mainly used in data science and machine learning research. It gives you acess to file browser (to upload, download, copy, rename, delete files), do data visualization, add data, code, texts, equations all in one place, use big data tools etc. It supports more than 40 programming languages and has an interactive output. It also allows you to share your work.

**Q. How does it work?**
  
You write your code or comments/text in rectangular “cells” and the browser then passes it to the back-end “kernel” which runs your code and returns output.

  
 > .ipynb file is a python notebook which stores code, text, markdown, plots, results in a specific format but .py file is a python file which only stores code and plain text (like comments etc).
  
  
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
  
`Note: package contains all the files you need for a module`  
  
1. Anaconda (or Conda): cross platform package and environment manager and may also access C, C++ libraries, R package for scientific computing along with Python.
2. Pip: tool to install Python software packages only. 

### Anaconda
- allows you to install softwares written in any programming language,
- flexibility to create different environments with different software versions,
- can use both CLI and GUI
  
> If you try to access a library with different version based on your project, pip may throw an error. To create isolated environments you can use virtual environment (venv) with pip.
  
## Deep Learning tutorial
  
* Request for resources
  
> Challenge 1
>
> 1. Go to mana.its.hawaii.edu and sign-in using your UH credentials  
> 2. Interactive Apps -> Jupyter Lab -> partition: gpu -> Time: 2 hours -> Nodes: 1 -> Tasks: 1 -> Cores: 2 -> RAM: 10 -> # of GPU: 1 -> GPU Type: any
> 3. Launch the session and open Jupyter Notebook
  
* Create a conda environment
* Download libraries

{% include links.md %}
