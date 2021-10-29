---
title: "Connecting to a remote HPC System"
teaching: 20
exercises: 0
questions:
- "How do I log in to a remote HPC system?"
- "What is Open OnDemand and how do I use it?"
objectives:
- "Understand how to connect to an HPC system"
- "Understand basics of Open OnDemand"
keypoints:
- "SSH is the main method of connecting to HPC systems"
- "Alternative tools like Open OnDemand exist to access HPC systems"
---

## Connecting to a Remote HPC system
The first step in using a cluster is to establish a connection from our laptop
to the cluster. When we are sitting at a computer (or standing, or holding it
in our hands or on our wrists), we have come to expect a visual display with
icons, widgets, and perhaps some windows or applications: a graphical user
interface, or GUI. Since computer clusters are remote resources that we connect
to over often slow or laggy interfaces (WiFi and VPNs especially), it is more
practical to use a command-line interface, or CLI, in which commands and
results are transmitted via text, only. Anything other than text (images, for
example) must be written to disk and opened with a separate program.

If you have ever opened the Windows Command Prompt or macOS Terminal, you have
seen a CLI. If you have already taken The Carpentries' courses on the UNIX
Shell or Version Control, you have used the CLI on your local machine somewhat
extensively. The only leap to be made here is to open a CLI on a *remote*
machine, while taking some precautions so that other folks on the network can't
see (or change) the commands you're running or the results the remote machine
sends back. We will use the Secure SHell protocol (or SSH) to open an encrypted
network connection between two machines, allowing you to send & receive text
and data without having to worry about prying eyes.

{% include figure.html url="" max-width="50%"
   file="/fig/connect-to-remote.svg"
   alt="Connect to cluster" caption="" %}


## Traditional HPC system access using Secure Shell (SSH)

Most modern computers have a built in SSH client to their terminal.
Alternative clients exist, primarily for windows or add-ons to web browsers, 
but they all operate in a similar manner. SSH clients are usually command-line tools, where you 
provide the remote machine address as the only required argument. 
If your username on the remote system differs from what
you use locally, you must provide that as well. If your SSH client has a
graphical front-end, such as PuTTY, MobaXterm, you will set these arguments
before clicking "connect." From the terminal, you'll write something like `ssh
userName@hostname`, where the "@" symbol is used to separate the two parts of a
single argument.

### Example of ssh on the Linux command line, Mac OS terminal and windows 10 terminal 
```
ssh dav@mana.its.hawaii.edu
```
{: .language-bash}

{% include figure.html url="" max-width="50%"
   file="/fig/CLI.png"
   alt="Connect to cluster" caption="" %}

> ## Take note
>
> You would replace `dav` with your username. 
> You may be asked for your password. Watch out: the
> characters you type after the password prompt are not displayed on the screen.
> Normal output will resume once you press `Enter`.
> 
{: .callout}

## Open OnDemand (OOD) - An alternative to using SSH

While, SSH is a common method to connect to remote systems (HPC or even servers), tools that provide
the same functionality and more exist.  One such tool is Open OnDemand (OOD).
> ## Learn more about OOD
>
> Created by Ohio Supercomputer Center, U. of Buffalo CCR, and Virginia Tech
> and development funded by National Science Foundation under 
> grant numbers 1534949 and 1835725. [Learn more about Open OnDemand](http://openondemand.org/)
>
{: .callout}

### Features of Open OnDemand

Open OnDemand works with a web browser making it possible to connect to an HPC system like Mana
with almost any device.  It has built in functionality for file browser, uploading and downloading 
smaller files, text editing, SSH terminal, and submitting interactive applications such as a Desktop 
on a compute node, Jupyter Lab and Rstudio.
> ### Interactive applications at other institutions
>
> Various other interactive applications have been made for Open OnDemand but are not available by default.
> See [here](https://osc.github.io/ood-documentation/master/install-ihpc-apps.html#) for a list of known interact appilications. 
>
{: .callout}
{% include figure.html url="" max-width="50%"
   file="/fig/MANA.png"
   alt="Connect to cluster" caption="" %}

> ## Login to Mana using Open onDemand
>
> > ## Browser choice and using an incognito or private browsing window
> > 
> > While almost any modern browser should work with OOD, the developers recommend google chrome as it has the widest support
> > for the tools used to create OOD [browser requirements](https://osc.github.io/ood-documentation/latest/requirements.html#browser-requirements)
> >
> > For security it is recommend you use a private browsing window with OOD as this allows a complete
> > log out by just simplying closing the window.  While logout does exist in OOD, it may not work as
> > expected and really keep you logged in even after hitting logout.
> >
> {:.callout}
> 
> * Open up your web browser and start a private browsing window.  Now, connect to the instance of Open OnDemand used with Mana by
> pointing your browser at [https://mana.its.hawaii.edu](https://mana.its.hawaii.edu). 
>
{: .challenge}

### file browsing and editing
The file browser allows you to perform directory manipulation, create new files, upload and download files without having to know the command line.
The file browser can even has the ability to do text editing on files 
which is useful if you are not familiar with a command line text editor.
> ## Command line text editors
>
> Common text editors you find on HPC systems or linx systems include:
> * [Vi/Vim](https://www.vim.org/)
> * [Emacs](https://www.gnu.org/software/emacs/)
> * [nano](https://www.nano-editor.org/)
> Of the three, nano is the simplest to use
>
{: .callout}
{% include figure.html url="" max-width="50%"
   file="/fig/ood_file_edit.png"
   alt="Connect to cluster" caption="" %}

 
### Terminal in the browser
As Open OnDemand doesn't really replace the traditional commandline/SSH access method to HPC systems,
and instead makes the use of certain applications simpler, it still provides a way to bring up a commandline
on an HPC system within your web browser.  
  {% include figure.html url="" max-width="50%"
   file="/fig/ood_shell.png"
   alt="Connect to cluster" caption="" %}



### Interactive applications
 While Open OnDemand can allow you to access HPC systems using the terminal, it also has the ability to expand the ways
 and HPC can be easily used though allowing the use of interactive applications that many have come to depend on.
  {% include figure.html url="" max-width="50%"
   file="/fig/ood_interact.png"
   alt="Connect to cluster" caption="" %}

Each application has a form which you use to define the resources your job requires so that Open OnDemand can submit it on your behalf.
It also has the ability to email you once your job starts as not all jobs will begin immediately.
  {% include figure.html url="" max-width="50%"
   file="/fig/ood_form.png"
   alt="Connect to cluster" caption="" %}

Finally, when a job begins, it presents you with a button you can click to start up your interactive application and use it within your 
browser.
  {% include figure.html url="" max-width="50%"
   file="/fig/ood_job.png"
   alt="Connect to cluster" caption="" %}

{% include links.md %}
