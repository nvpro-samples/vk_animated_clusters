# Changelog for vk_animated_clusters

* 2025/2/11:
  * Expose more cluster config options for nvidia cluster builder library.
  * Add option for using per-cluster vertices. Note, this increases memory quite a bit, as well as animation processing. And can be useful to have more metric to compare with.
* 2025/2/4: `doAnimation` is moved to renderer config. This allows a newly added codepath for the triangle ray tracer to use BLAS compaction when animation is off. It enables more comparisons between ray traced triangles and clusters.
* 2025/1/30: Initial Release