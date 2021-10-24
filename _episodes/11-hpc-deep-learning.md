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

# Connecting to UH-HPC cluster Mana (SSH vs OOD)

On logging into a cluster, you are into a login node which has limited storage and restricted use because it is being shared by everybody so you submit your jobs to a compute node to get required resources for your research work. These compute nodes have lots of storage. There are 2 ways to access the cluster:
1. Using Secure Shell (SSH)
   - done using Linux like commands in terminal (alo known as Command Line Interface or CLI),
   - login using **ssh <username>@mana.its.hawaii.edu**,
   - works with weak connections too.
  
2. Open On Demand (OOD)
   - a web based interface (or graphical user interface (GUI),
   - easy to access,
   - use interactive apps like Jupyter, RStudio and even Linux shell,
   - easy access to file systems as well.
  
## Jupyter Lab vs Jupyter Notebook

 Jupyter notebook allows you to access only .ipynb files, i.e. it will create a computational environment which stores your code, results, plots, texts etc. And here you can work only in one of your environment. But Jupyter Lab gives a better user interface along with all the facilties provided by the notebook. It has a modular structure where one can access .py, .ipynb, html or markdown files, access filebrowser, all in the same window. 
  
 > .ipynb file is a python notebook which stores code, text, markdown, plots, results in a specific format but .py file is a python file which only stores code and plain text (like comments etc).



{% include links.md %}
