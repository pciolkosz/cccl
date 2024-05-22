Level types
================================================================================

.. _hierarchy_levels:

Each hierarchy level has a type to represent it. Toegether with extents object 
it is used to build :cpp:struct:`cuda::experimental::level_dimensions`, which describes
dimensions of a single level in the hierarchy. While you can use the level types
directly to build that description, it is recommended to use helper functions described
:ref:`here <level_dimensions>`.
Level types can also be used in hierarchy queries when you need to point to a certain level,
for example when asking about a thread count on a specific hierarchy level.
The details are described :cpp:struct:`here <cuda::experimental::hierarchy_dimensions_fragment>`.

Currently available levels:

* :cpp:struct:`cuda::experimental::grid_level`
* :cpp:struct:`cuda::experimental::cluster_level`
* :cpp:struct:`cuda::experimental::block_level`
* :cpp:struct:`cuda::experimental::thread_level`

For convinience, each level type has a constexpr instance variable:

* :cpp:member:`cuda::experimental::grid`
* :cpp:member:`cuda::experimental::cluster`
* :cpp:member:`cuda::experimental::block`
* :cpp:member:`cuda::experimental::thread`