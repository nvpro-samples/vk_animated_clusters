# Changelog for vk_animated_clusters
* 2026-10-26:
  * HBAO runs at quarter resolution by default (`--hbaoquarter 0` for full), upsampled depth-aware.
  * HBAO normals are rebuilt per pixel from the plane-consistent neighbours, which removes the dark dashes along creases.
* 2026-10-22:
  * Fix `--addcamerapath` appending to the paths loaded from the scene's `.camerapaths.txt`, the first one now replaces them.
  * More command line / sequence parameters: build modes (`--templatebuildmode`, `--templateinstantiatemode`, `--clusterbuildmode`, `--blasbuildmode`), `--templatebboxbloat`, `--positiontruncatebits`, `--blasrebuildfraction`, `--tlasrebuild`, shading (`--facetshading`, `--visualize`, `--shadow`, `--aorays`, `--aoradius`, `--hbaointensity`, ...) and deformers (`--ripple*`, `--twist*`). `--autosettings` (default on) derives the build modes and `--templates` from the animation state at startup and when it changes, `--autosettings 0` keeps the given values.
  * Use the reversed-Z depth buffer and the HBAO of `vk_lod_clusters`, it always runs at full resolution now (`--hbaoblur`, `--hbaoradius`).
  * The viewport keeps the aspect ratio of the fixed resolutions (letterboxed), they are displayed and saved as is instead of being scaled to the window.
  * `--supersample` takes fixed resolutions (540, 720, 1080, 1440, 2160 and square 1024, 2048, 4096), `--windowsize` replaces `--winsize`.
  * Headless runs step the animation by a fixed time (`--fixedtimestep`), screenshots are repeatable.
  * clang-format and copyright years.
  * Add the camera path system of `vk_lod_clusters` (`--addcamerapath`, `--loadcamerapaths`, `--runcamerapath`), see `docs/camera_paths.md`.
  * Parameter sequences log a `MemoryReport` per sequence with the memory usage and scene counts.
  * Add `--headless`, parameter sequences (`--sequencefile`), screenshots (`--sequencescreenshot`, `--screenshotframegap`, F12) and `tests/run_tests.py`.
  * Switch `LICENSE`, `CONTRIBUTING` and the file headers to the ones of `vk_lod_clusters`, `background.*.glsl` listed the proprietary license.
* 2025-12-9:
  * Removed meshoptimizer as submodule, it is now part of nvpro_core2 
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