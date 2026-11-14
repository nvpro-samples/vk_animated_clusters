# Tests

Every test is a **sequence file** that walks a set of configurations, the sample re-initializes the
renderer whenever one of them changes. The sequences are run headless, so no window is required.
There is no golden image comparison: the question is whether every configuration still builds its
pipelines and renders. A screenshot is written at the end of each sequence so a failure can be
looked at.

``` bash
# runs every sequence file in this directory
python tests/run_tests.py

# only the ones whose filename contains "hbao", and show the sample's output
python tests/run_tests.py --filter hbao --verbose

# against a different build or scene
python tests/run_tests.py --exe _bin/Debug/vk_animated_clusters.exe --scene path/to/scene.gltf
```

Without `--exe` the newer binary of `_bin/Release` and `_bin/RelWithDebInfo` is tested, `_bin/Debug` only
when neither exists. The log and `screenshot_<index>_<sequence name>.jpg` of every file end up in
`tests/_results/<timestamp>/<sequence file>/` (gitignored).

A test fails when the sample reports a validation error, a shader compilation error, a failed
renderer init, a lost device, when it produced fewer screenshots than it has sequences, or when it
doesn't exit cleanly. The `MemoryReport` block after each sequence lists the memory per category (the
BLAS and CLAS bytes are what the build actually used) and the scene's cluster, triangle and instance
counts. The runner doesn't judge the numbers, they are in `output.log` and `--verbose` prints them. The Vulkan validation layers are enabled by default, use `--no-validation` to
skip them, `--no-screenshots` to skip the images. Everything after the known options is passed on to
the sample, so `python tests/run_tests.py -- --device 1` picks another GPU.

## The sequence files

| file | what it covers |
|------|----------------|
| [renderers.txt](renderers.txt)                   | the four renderers, with and without animation, including switching between them |
| [templates.txt](templates.txt)                   | cluster ray tracing with explicit, implicit and without templates |
| [clusterconfigs.txt](clusterconfigs.txt)         | every cluster size on the raster and ray tracing cluster renderers |
| [scene_copies.txt](scene_copies.txt)             | scene instancing, which scales the cluster, template and BLAS counts |
| [camerapath.txt](camerapath.txt)                 | a fly-through along a camera path (`--runcamerapath`) with each renderer, animated and static |
| [hbao_supersample.txt](hbao_supersample.txt)     | ambient occlusion settings against the reversed-Z depth buffer, crossed with the render resolution |
| [build_flags.txt](build_flags.txt)               | acceleration structure build modes (`--templatebuildmode`, `--clusterbuildmode`, `--blasbuildmode`, ...) crossed with templates and animation, TLAS rebuild, position truncation, template bbox bloat; the `build flags` log line shows what was applied |
| [shading_animation.txt](shading_animation.txt)   | visualization, facet shading, shadow and ambient occlusion rays, the ripple and twist deformers, time override, draw toggles, and all numeric values at their extremes |

Adding a test means adding a `.txt` file here, the runner picks it up automatically. The syntax
is the one of the parameter sequencer, `#` starts a comment:

```
SEQUENCE "name of the sequence"  --sequenceframes 8 --sequenceresetframes 2 --renderer 1
SEQUENCE "the next one"          --clusterconfig 2
```

Headless runs step the animation by a fixed 1/60 s per frame (`--fixedtimestep`), so the screenshots are repeatable.

Every sequence inherits the state of the previous one, only name the parameters that change.
Use `--help` on the sample for the full list of parameters.
