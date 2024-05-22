CUDA Experimental
==================================================

.. toctree::
   :hidden:
   :maxdepth: 2

   thread_hierarchy_description/thread_hierarchy_description
   ${repo_docs_api_path}/CUDAX_api

.. the line below can be used to use the README.md file as the index page
.. .. mdinclude:: ../README.md

What is CUDA Experimental?
**************************************************

CUDA Experimental serves as a distribution channel for features that are considered experimental in the CUDA C++ Core Libraries. Some of them are still actively designed or developed and their API is evolving.
Some of them are specific to one hardware architecture and are still looking for a generic and forward compatible exposure. Finally, some of them need to prove useful enough to deserve long term support.

All APIs available in CUDA Experimental are not considered stable and can change without a notice. They can also be deprecated or removed on a much faster cadence than in other CCCL libraries.

Features are exposed here for the CUDA C++ community to experiment with and provide feedback on how to shape it to best fit their use cases.
Once we become confident a feature is ready and would be a great permanent addition in CCCL, it will become a part of some other CCCL library with a stable API.

Projects available in CUDA Experimental
**************************************************

* :ref:`Thread Hierarchy Description <thread_hierarchy_description>`

  * Describe a hierarchy of CUDA threads with a mix of static and dynamic information
  * Count and enumerate CUDA threads