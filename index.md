---
layout: lesson
root: .
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---

{% include gh_variables.html %}

This workshop focuses on Utilization of High performance Computing (HPC) clusters, such as Mana, for deep learning tasks as well as the benefits of understanding the different types of file systems available on HPC clusters and basic ways one would stage data from slower file systems to faster file systems.
Attendees will learn how to use Jupyter Lab on Mana via Open OnDemand, the benefits of using GPUs over just CPUs for deep learning applications and how staging data the correct file system can improve performance of deep learning jobs.


> ## Prerequisites
>
> * Command line experience is necessary for this lesson. We recommend the  participants to go through [shell-novice](https://swcarpentry.github.io/shell-novice/), if new to > > 
> * Have an account on Mana 
> * Have UH Duo/MFA enabled
> * Be able to connect to the workshop in Zoom
> * Have a modern web browser
{: .prereq}

By the end of this workshop, students will know how to:

* Have a basic understanding of how to access Open OnDemand and use Jupyter Lab on Mana
* Have a basic understanding of how to request GPUs and utilize them for deep learning tasks
* Have a basic understanding of the performance difference between CPU and GPU for deep learning
* Have a basic understanding of different types of file system found on HPC clusters
* Have a basic understanding of how correct file system choice affect deep learning performance


> ## Getting Started
>
> To get started, follow the directions in the "[Setup](
> {{ page.root }}/setup.html)" tab to download data to your computer and follow
> any installation instructions.
{: .callout}


<!-- > ## For Instructors -->
<!-- > -->
<!-- > If you are teaching this lesson in a workshop, please see the -->
<!-- > [Instructor notes](guide/). -->
<!-- {: .callout} -->

{% include links.md %}
