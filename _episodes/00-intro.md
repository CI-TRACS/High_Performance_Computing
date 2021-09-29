---
title: "Introduction"
teaching: 15
exercises: 5
questions:
- "What are different types of computational resouces?"
- "What are different types of computational tools available?"
- "What can I expect to learn from this workshop?"
objectives:
- "Identify different tools of advance cyberinfrastructure"
- "Know when to use a High Performance Computing Cluster vs. cloud computing."
keypoints:
- "High Performance Computing (HPC) typically involves connecting to very large computing systems elsewhere in the world."
- "Cloud computing resources, such as JetStream2, allow for persistent services, but may not be as well suited for computationally intesnsive workloads."
- "Tapis can be used to join both HPC and cloud computing resources to simplify workflows."
---

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

## When you need persistent services


## When you need to tie resources together



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