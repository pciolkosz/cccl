Thread hierarchy description
================================================================================

.. toctree::
   :glob:
   :hidden:

   hierarchy_levels
   level_dimensions
   Hierarchy dimensions <../${repo_docs_api_path}/structcuda_1_1experimental_1_1hierarchy__dimensions__fragment>

..
   I have no idea why the .. in the reference above is needed

.. _thread_hierarchy_description:

Thread hierarchy APIs in :cpp:`cuda::experimental` namespace allow to describe dimensions of 
a hierarchy of CUDA threads as a mix of static an dynamic values.
The hierarchy can later be used to count or enumerate threads. It allows to
conviniently wrap calculations that otherwise require complex expressions involving
a mix of `threadIdx`, `blockIdx`, `blockDim`, `gridDim` etc. Static information
contained in the hierarchy type allows to optimize these calculation or provide
constexpr values to scale resources like shared memory.

Hierarchy creation
*******************************************************************************

Hierarchy type can be created using :cpp:func:`cuda::experimental::make_hierarchy` function.
It takes a variable number of arguments describing dimensions of the hierarchy
levels. Description of the levels can be created using `*_dims` functions
described :ref:`here <level_dimensions>`.

.. code-block:: c++
   :caption: Example creating a simple hierarchy
   :name: simple_hierarchy_creation

   using namespace cuda::experimental;

   // Describe grid dimensions as dynamic 256 on x dimension and static 1 on y and z
   auto grid_dimensions = grid_dims(256);
   // Describe block dimensions to statically be 8 along all 3 dimensions
   auto block_dimensions = block_dims<8, 8, 8>();
   auto hierarchy = make_hierarchy(grid_dimensions, block_dimensions);

Hierarchy queries
********************************************************************************

Once the hierarchy is created, it can be used to provide query infomation about it like
count threads or provide indexing in the device code. Each query API takes two arguments,
unit and level. Both of them can be expressed with hierarchy level types described
:ref:`here <hierarchy_levels>`.

.. code-block:: c++
   :caption: Example hierarchy queries
   :name: simple_hierarchy_queries

   using namespace cuda::experimental;

   auto hierarchy = make_hierarchy(grid_dims(256), block_dims<8, 8, 8>());
   // Count threads (unit) in block (level)
   // It uses only static information, so the result is available at compile time
   static_assert(hierarchy.count(thread, block) == 8 * 8 * 8);
   
   // level is defaulted to the top-most level in the hierarchy (grid_level in this case)
   assert(hierarchy.extents(block).extent(0) == 256);

   // Device code only
   assert(hierarchy.index(block).x == blockIdx.x);

More details about the hierarchy creation and usage can be found :cpp:struct:`here <cuda::experimental::hierarchy_dimensions_fragment>`.