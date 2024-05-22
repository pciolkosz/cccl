Level dimensions
================================================================================

.. _level_dimensions:

:cpp:struct:`cuda::experimental::level_dimensions` type template can be used to
describe dimensions of a level in :cpp:type:`cuda::experimental::hierarchy_dimensions`.
Unit for the dimensions is implied by the next level in the hierarchy or its thread
in case of the level at the bottom.
In order to create instances of that type it is recommended to use the following helper functions:

* :cpp:func:`cuda::experimental::grid_dims` creates dimensions marked with :cpp:struct:`cuda::experimental::grid_level`
* :cpp:func:`cuda::experimental::cluster_dims` creates dimensions marked with :cpp:struct:`cuda::experimental::cluster_level`
* :cpp:func:`cuda::experimental::block_dims` creates dimensions marked with :cpp:struct:`cuda::experimental::block_level`

Each of the helper function accept integral or `dim3` types. Each function also has
an argument-less overload that accepts up to three `unsigned int` template arguments
to describe the dimensions statically.