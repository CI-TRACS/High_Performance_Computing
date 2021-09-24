---
title: "Introduction"
teaching: 15
exercises: 5
questions:
- "Why would I be interested in High Performance Computing (HPC)?"
- "What can I expect to learn from this course?"
objectives:
- "Identify how an HPC system could benefit you."
keypoints:
- "High Performance Computing (HPC) typically involves connecting to very large computing systems elsewhere in the world."
- "These other systems can be used to do work that would either be impossible or much slower on smaller systems."
---

Frequently, research problems that use computing can outgrow the capabilities
of the desktop or laptop computer where they started:

* A statistics student wants to cross-validate a model. This involves running
  the model 1000 times &mdash; but each run takes an hour. Running the model on
  a laptop will take over a month! In this research problem, final results are
  calculated after all 1000 models have run, but typically only one model is
  run at a time (in **serial**) on the laptop. Since each of the 1000 runs is
  independent of all others, and given enough computers, it's theoretically
  possible to run them all at once (in **parallel**).
* A genomics researcher has been using small datasets of sequence data, but
  soon will be receiving a new type of sequencing data that is 10 times as
  large. It's already challenging to open the datasets on a computer &mdash;
  analyzing these larger datasets will probably crash it. In this research
  problem, the calculations required might be impossible to parallelize, but a
  computer with **more memory** would be required to analyze the much larger
  future data set.
* An engineer is using a fluid dynamics package that has an option to run in
  parallel. So far, this option was not used on a desktop. In going from 2D
  to 3D simulations, the simulation time has more than tripled. It might be
  useful to take advantage of that option or feature. In this research problem,
  the calculations in each region of the simulation are largely independent of
  calculations in other regions of the simulation. It's possible to run each
  region's calculations simultaneously (in **parallel**), communicate selected
  results to adjacent regions as needed, and repeat the calculations to
  converge on a final set of results. In moving from a 2D to a 3D model, **both
  the amount of data and the amount of calculations increases greatly**, and
  it's theoretically possible to distribute the calculations across multiple
  computers communicating over a shared network.

In all these cases, access to more (and larger) computers is needed. Those
computers should be usable at the same time, **solving many researchers'
problems in parallel**.

> ## Break the Ice
>
> Talk to your neighbour, office mate or [rubber
> duck](https://rubberduckdebugging.com/) about your research.
>
> * How does computing help you do your research?
> * How could more computing help you do more or better research?
{: .discussion }


## When Tasks Take Too Long

When the task to solve becomes heavy on computations, the operations are
typically out-sourced from the local laptop or desktop to elsewhere. Take for
example the task to find the directions for your next vacation. The
capabilities of your local machine are typically not enough to calculate that route
spontaneously: [finding the shortest path](
https://en.wikipedia.org/wiki/Dijkstra's_algorithm. Instead of doing this yourself, you use a website,
which in turn runs on a server, that is almost definitely not in the same room
as you are.

While the above task can be dealt with on a single server, larger computationally intensive 
task or analysis may become too daunting for a single server to complete.  To solve these larger problems,
larger agglomerations of servers are used. These go by the name of
"clusters" or "super computers".

The methodology of providing the input data, configuring the program options,
and retrieving the results is quite different to using a local machine.
Moreover, while tools like [Open OnDemand](https://openondemand.org/) can provide a middle ground with some amount of graphical access,
in many cases the graphical interface is discarded in favor of using the
command line. This imposes a double paradigm shift for prospective users asked
to

1. work with the command line interface (CLI), rather than a graphical user
   interface (GUI)
1. work with a distributed set of computers (called nodes) rather than the
   machine attached to their keyboard & mouse

> ## I've Never Used a Server, Have I?
>
> Take a minute and think about which of your daily interactions with a
> computer may require a remote server or even cluster to provide you with
> results.
>
> > ## Some Ideas
> >
> > * Checking email: your computer (possibly in your pocket) contacts a remote
> >   machine, authenticates, and downloads a list of new messages; it also
> >   uploads changes to message status, such as whether you read, marked as
> >   junk, or deleted the message. Since yours is not the only account, the
> >   mail server is probably one of many in a data center.
> > * Searching for a phrase online involves comparing your search term against
> >   a massive database of all known sites, looking for matches. This "query"
> >   operation can be straightforward, but building that database is a
> >   [monumental task](https://en.wikipedia.org/wiki/MapReduce)! Servers are
> >   involved at every step.
> > * Searching for directions on a mapping website involves connecting your
> >   (A) starting and (B) end points by [traversing a graph](
> >   https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) in search of
> >   the "shortest" path by distance, time, expense, or another metric.
> >   Converting a map into the right form is relatively simple, but
> >   calculating all the possible routes between A and B is expensive.
> >
> > Checking email could be serial: your machine connects to one server and
> > exchanges data. Searching by querying the database for your search term (or
> > endpoints) could also be serial, in that one machine receives your query
> > and returns the result. However, assembling and storing the full database
> > is far beyond the capability of any one machine. Therefore, these functions
> > are served in parallel by a large, ["hyperscale"](
> > https://en.wikipedia.org/wiki/Hyperscale_computing) collection of servers
> > working together.
> {: .solution}
{: .challenge }

{% include links.md %}