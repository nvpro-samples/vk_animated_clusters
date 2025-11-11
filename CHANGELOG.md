# Changelog for vk_animated_clusters
* 2025-11-11:
  * Removed deprecated `nv_cluster_builder` library
  * Replace triangle stripification with `meshopt_optimizeMeshlet` and always run it
* 2025-8-24:
  * Updated `meshoptimizer` submodule to `v 0.25`
* 2025-7-7:
  * Fix regression with sky rendering & missing sky UI
* 2025-6-26:
  * Updated `nv_cluster_library` submodule with new API and proper vertex limit support.
* 2025-6-23:
  * Updated to use `nvpro_core2`, as result command-line arguments are now prefixed with `--` rather than just `-`. It is recommended to delete existing `/_build` or the CMake cache prior building or generating new solutions.
* 2025-6-17:
  * Updated `meshoptimizer` submodule to `v 0.24` using `meshopt_buildMeshletsSpatial` for improved ray tracing clusterization.
* 2025-5-28:
  * Reduce memory consumption when animation is disabled: allocate normals, clas and blas only once per unique geometry.
* 2025-4-30:
  * Automatically set preferred ray tracing build settings when animation is toggled on or off in UI.
  * Highlight render resolution in UI when ray tracing.
* 2025-4-25:
  * bugfix gltf loading of meshes with multiple primitives
* 2025-2-11:
  * Expose more cluster config options for nvidia cluster builder library.
  * Add option for using per-cluster vertices. Note, this increases memory quite a bit, as well as animation processing. And can be useful to have more metric to compare with.
* 2025-2-4: `doAnimation` is moved to renderer config. This allows a newly added codepath for the triangle ray tracer to use BLAS compaction when animation is off. It enables more comparisons between ray traced triangles and clusters.
* 2025-1-30: Initial Release