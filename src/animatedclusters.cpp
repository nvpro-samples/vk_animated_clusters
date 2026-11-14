/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <nvutils/file_operations.hpp>
#include <nvgui/camera.hpp>

#include "animatedclusters.hpp"

bool g_verbose = false;

namespace animatedclusters {

AnimatedClusters::AnimatedClusters(const Info& info)
    : m_info(info)
{
  nvutils::ProfilerTimeline::CreateInfo createInfo;
  createInfo.name = "graphics";

  m_profilerTimeline = m_info.profilerManager->createTimeline(createInfo);

  registerCameraPathParameters();

  m_screenshotTimeStamp =
      fmt::format("{:%Y_%m_%d_%H_%M_%S}", std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now()));

  m_info.parameterRegistry->add({"scene"}, {".gltf", ".glb"}, &m_sceneFilePath);
  m_info.parameterRegistry->add({"sequencescreenshot", "save screenshot at end of each sequence. 0 disabled (default), 1 full window, 2 rendered viewport"},
                                (int*)&m_sequenceScreenshotMode, true);
  m_info.parameterRegistry->add({"screenshotframegap", "save a screenshot every n frames, and after the last headless frame. 0 disabled (default)"},
                                &m_screenshotFrameInterval);
  m_info.parameterRegistry->add({"screenshotmode", "screenshot content of screenshotframegap: 1 full window, 2 rendered viewport (default)"},
                                (int*)&m_screenshotMode);
  m_info.parameterRegistry->add({"fixedtimestep", "seconds the animation advances per frame, 0 uses the real time. default 1/60 when headless, otherwise 0"},
                                &m_fixedTimeStep);
  m_info.parameterRegistry->add({"hbao", "ambient occlusion in rasterization (HBAO). default true"}, &m_tweak.hbaoActive);
  m_info.parameterRegistry->add({"hbaoblur", "blur the HBAO result. default true"}, &m_frameConfig.hbaoSettings.blur);
  m_info.parameterRegistry->add({"hbaoquarter", "HBAO at half width and height, upsampled depth-aware. default true"},
                                &m_tweak.hbaoQuarter);
  m_info.parameterRegistry->add({"hbaoradius", "HBAO radius as percentage of the scene bounding radius. default 0.05"},
                                &m_tweak.hbaoRadius);
  m_info.parameterRegistry->add({"renderer"}, (int*)&m_tweak.renderer);
  m_info.parameterRegistry->add({"verbose"}, &g_verbose, true);
  m_info.parameterRegistry->add({"resetstats"}, &m_tweak.autoResetTimers);
  m_info.parameterRegistry->add({"supersample", "1 to 4 times the window size, or fixed 540, 720, 1080, 1440, 2160 and square 1024, 2048, 4096"},
                                &m_tweak.supersample);
  m_info.parameterRegistry->add({"animation"}, &m_rendererConfig.doAnimation);
  m_info.parameterRegistry->add({"templatebboxbloat", "template bbox bloat percentage, negative disables passing the bbox. default 0.5"},
                                &m_tweak.templateBboxBloat, -0.001f, 1.0f);
  m_info.parameterRegistry->add({"positiontruncatebits", "cluster position truncation bits, 0 to 22. default 0"},
                                &m_tweak.clusterPositionTruncationBits, 0u, 22u);
  m_info.parameterRegistry->add({"blasrebuildfraction", "fraction of the BLAS rebuilt per frame by the triangle ray tracer when animating, 0 to 1. default 0.1"},
                                &m_frameConfig.blasRebuildFraction, 0.f, 1.f);
  m_info.parameterRegistry->add({"tlasrebuild", "always fully rebuild the TLAS instead of refitting it. default false"},
                                &m_frameConfig.forceTlasFullRebuild);

  m_info.parameterRegistry->add({"facetshading", "facet instead of smooth normals. default true"}, &m_tweak.facetShading);
  m_info.parameterRegistry->add({"visualize", "0 none, 1 clusters, 2 triangles. default 0"},
                                &m_frameConfig.frameConstants.visualize, 0u, 2u);
  m_info.parameterRegistry->add({"drawobjects", "draw the scene objects. default true"}, &m_frameConfig.drawObjects);
  m_info.parameterRegistry->add({"clusterboxes", "draw the static cluster bboxes, raster only. default false"},
                                &m_frameConfig.drawClusterBoxes);
  m_info.parameterRegistry->add({"colorxor", "xor value for the cluster and triangle visualization colors. default 0"},
                                &m_frameConfig.frameConstants.colorXor);
  m_info.parameterRegistry->add({"lightmixer", "mix between the two lights, 0 to 1. default 0.5"},
                                &m_frameConfig.frameConstants.lightMixer, 0.f, 1.f);
  m_info.parameterRegistry->add({"shadow", "ray traced shadows, 0 or 1. default 1"},
                                &m_frameConfig.frameConstants.doShadow, 0u, 1u);
  m_info.parameterRegistry->add({"aorays", "ray traced ambient occlusion rays, 0 to 64. default 2"},
                                &m_frameConfig.frameConstants.ambientOcclusionRays, 0, 64);
  m_info.parameterRegistry->add({"aoradius", "ray traced ambient occlusion radius. default 0.1"},
                                &m_frameConfig.frameConstants.ambientOcclusionRadius, 0.001f, 1.f);
  m_info.parameterRegistry->add({"hbaointensity", "HBAO intensity. default 1"}, &m_frameConfig.hbaoSettings.intensity);
  m_info.parameterRegistry->add({"hbaobias", "HBAO bias. default 0.1"}, &m_frameConfig.hbaoSettings.bias);
  m_info.parameterRegistry->add({"hbaopower", "HBAO power exponent. default 4"}, &m_frameConfig.hbaoSettings.powerExponent);

  m_info.parameterRegistry->add({"ripple", "ripple deformer, 0 or 1. default 1"},
                                &m_frameConfig.frameConstants.animationRippleEnabled, 0, 1);
  m_info.parameterRegistry->add({"ripplefrequency", "ripple frequency. default 50"},
                                &m_frameConfig.frameConstants.animationRippleFrequency, 0.001f, 200.f);
  m_info.parameterRegistry->add({"rippleamplitude", "ripple amplitude. default 0.004"},
                                &m_frameConfig.frameConstants.animationRippleAmplitude, 0.f, 0.05f);
  m_info.parameterRegistry->add({"ripplespeed", "ripple speed. default 5"},
                                &m_frameConfig.frameConstants.animationRippleSpeed, 0.f, 10.f);
  m_info.parameterRegistry->add({"twist", "twist deformer, 0 or 1. default 1"},
                                &m_frameConfig.frameConstants.animationTwistEnabled, 0, 1);
  m_info.parameterRegistry->add({"twistangle", "twist maximum angle in degrees. default 180"},
                                &m_frameConfig.frameConstants.animationTwistMaxAngle, 0.f, 1440.f);
  m_info.parameterRegistry->add({"twistspeed", "twist speed. default 0.1"},
                                &m_frameConfig.frameConstants.animationTwistSpeed, 0.f, 2.f);
  m_info.parameterRegistry->add({"autosettings", "derive the animation dependent build modes and templates automatically, 0 keeps whatever is set. default true"},
                                &m_autoSettings);
  m_info.parameterRegistry->add({"templates", "use cluster templates instead of dedicated CLAS builds. default depends on animation"},
                                &m_tweak.useTemplates);
  m_info.parameterRegistry->add({"templatebuildmode", "template build mode: 0 default, 1 fast build, 2 fast trace"},
                                (int32_t*)&m_tweak.templateBuildMode, 0, 2);
  m_info.parameterRegistry->add({"templateinstantiatemode",
                                 "template instantiate mode: 0 default, 1 fast build, 2 fast trace. default depends on animation"},
                                (int32_t*)&m_tweak.templateInstantiateMode, 0, 2);
  m_info.parameterRegistry->add({"clusterbuildmode", "CLAS build mode without templates: 0 default, 1 fast build, 2 fast trace. default depends on animation"},
                                (int32_t*)&m_tweak.clusterBuildMode, 0, 2);
  m_info.parameterRegistry->add({"blasbuildmode", "BLAS build mode, also for the triangle ray tracer: 0 default, 1 fast build, 2 fast trace. default depends on animation"},
                                (int32_t*)&m_tweak.blasBuildMode, 0, 2);
  m_info.parameterRegistry->add({"implicittemplates"}, &m_tweak.useImplicitTemplates);
  m_info.parameterRegistry->add({"gridcopies"}, &m_tweak.gridCopies);
  m_info.parameterRegistry->add({"gridconfig"}, &m_tweak.gridConfig);
  m_info.parameterRegistry->add({"clusterconfig"}, (int*)&m_tweak.clusterConfig);
  m_info.parameterRegistry->add({"clusterdedicatedvertices"}, &m_sceneConfig.clusterDedicatedVertices);
  m_info.parameterRegistry->add({"meshoptspatial"}, &m_sceneConfig.clusterSpatial);
  m_info.parameterRegistry->add({"meshoptspatialfill"}, &m_sceneConfig.clusterMeshoptSpatialFill);
  m_info.parameterRegistry->add({"meshoptflexcone"}, &m_sceneConfig.clusterMeshoptFlexCone);
  m_info.parameterRegistry->add({"meshoptflexsplit"}, &m_sceneConfig.clusterMeshoptFlexSplit);
  m_info.parameterRegistry->add({"moclusterspatialfill"}, &m_sceneConfig.clusterMeshoptSpatialFill);
  m_info.parameterRegistry->add({"overridetime"}, &m_tweak.overrideTime);
  m_info.parameterRegistry->add({"processingthreadpct", "float percentage of threads during initial file load and processing into lod clusters, default 0.5 == 50 %"},
                                &m_sceneConfig.processingThreadsPct);

  m_frameConfig.frameConstants                        = {};
  m_frameConfig.frameConstants.ambientOcclusionRays   = 2;
  m_frameConfig.frameConstants.facetShading           = 1;
  m_frameConfig.frameConstants.doShadow               = 1;
  m_frameConfig.frameConstants.ambientOcclusionRadius = 0.1f;
  m_frameConfig.frameConstants.lightMixer             = 0.5f;

  m_frameConfig.frameConstants.animationRippleEnabled   = 1;
  m_frameConfig.frameConstants.animationRippleFrequency = 50.f;
  m_frameConfig.frameConstants.animationRippleAmplitude = 0.004f;
  m_frameConfig.frameConstants.animationRippleSpeed     = 5.f;
  m_frameConfig.frameConstants.animationTwistEnabled    = 1;
  m_frameConfig.frameConstants.animationTwistSpeed      = 0.1f;
  m_frameConfig.frameConstants.animationTwistMaxAngle   = 180.f;
}

bool AnimatedClusters::initScene(const std::filesystem::path& filePath)
{
  deinitScene();

  std::string fileName = nvutils::utf8FromPath(filePath);

  LOGI("Loading scene: %s\n", fileName.c_str());

  m_scene = std::make_unique<Scene>();
  if(!m_scene->init(filePath, m_sceneConfig, m_resources))
  {
    LOGW("Loading scene failed\n");

    m_scene = nullptr;
    return false;
  }

  adjustSceneClusterConfig();
  m_sceneFilePath = filePath;

  // explicitly provided paths (--addcamerapath/--loadcamerapaths) are global and take precedence over the scene's file
  if(!m_cameraPathsExternal)
  {
    m_cameraPaths.clear();
    m_cameraPathActive   = -1;
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;

    std::filesystem::path pathsFile = sceneCameraPathsFile(m_sceneFilePath);
    if(std::filesystem::exists(pathsFile) && loadCameraPaths(pathsFile, m_cameraPaths))
    {
      m_cameraPathActive = m_cameraPaths.empty() ? -1 : 0;
      LOGI("Loaded %zu camera path(s) from %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(pathsFile).c_str());
    }
  }

  return true;
}

void AnimatedClusters::deinitScene()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_scene)
  {
    m_scene->deinit(m_resources);
    m_scene = nullptr;
  }
}

void AnimatedClusters::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_windowSize = size;
  m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoQuarter);
  updateImguiImage();
  if(m_renderer)
  {
    m_renderer->updatedFrameBuffer(m_resources);
  }
}

void AnimatedClusters::updateImguiImage()
{
  if(m_imguiTexture)
  {
    ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);
    m_imguiTexture = nullptr;
  }

  VkImageView imageView = m_resources.m_frameBuffer.useResolved ? m_resources.m_frameBuffer.imgColorResolved.descriptor.imageView :
                                                                  m_resources.m_frameBuffer.imgColor.descriptor.imageView;

  assert(imageView);

  m_imguiTexture = ImGui_ImplVulkan_AddTexture(m_imguiSampler, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

const AnimatedClusters::ClusterInfo AnimatedClusters::s_clusterInfos[NUM_CLUSTER_CONFIGS] = {
    {64, 64, CLUSTER_64T_64V},     {64, 128, CLUSTER_64T_128V},   {64, 192, CLUSTER_64T_192V},
    {96, 96, CLUSTER_96T_96V},     {128, 128, CLUSTER_128T_128V}, {128, 256, CLUSTER_128T_256V},
    {256, 256, CLUSTER_256T_256V},
};

void AnimatedClusters::adjustSceneClusterConfig()
{
#if 0
  m_scene->m_clusterTriangleHistogram.resize(m_scene->m_maxClusterTriangles + 1);
  m_scene->m_clusterVertexHistogram.resize(m_scene->m_maxClusterVertices + 1);

  // adjust the ui and settings based on actual data
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    const ClusterInfo& entry = s_clusterInfos[i];
    if(m_scene->m_maxClusterTriangles <= entry.tris && m_scene->m_maxClusterVertices <= entry.verts)
    {
      m_tweak.clusterConfig = entry.cfg;

      m_scene->m_config.clusterTriangles = entry.tris;
      m_sceneConfig.clusterTriangles     = entry.tris;
      m_sceneConfigLast.clusterTriangles = entry.tris;

      m_scene->m_config.clusterVertices = entry.verts;
      m_sceneConfig.clusterVertices     = entry.verts;
      m_sceneConfigLast.clusterVertices = entry.verts;
      return;
    }
  }
#endif
}

void AnimatedClusters::updatedClusterConfig()
{
  for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
  {
    if(s_clusterInfos[i].cfg == m_tweak.clusterConfig)
    {
      m_sceneConfig.clusterTriangles = s_clusterInfos[i].tris;
      m_sceneConfig.clusterVertices  = s_clusterInfos[i].verts;
      return;
    }
  }
}

void AnimatedClusters::onPreRender()
{
  m_profilerTimeline->frameAdvance();

  if(m_screenshotFrameInterval > 0 && m_frames > 0 && (m_frames % m_screenshotFrameInterval) == 0)
  {
    saveScreenshot(m_screenshotMode, fmt::format("screenshot_{}_{:08}.jpg", m_screenshotTimeStamp, m_frames - 1));
  }
}

void AnimatedClusters::onLastHeadlessFrame()
{
  // the interval never covers the final frame, which is typically the one of interest
  if(m_screenshotFrameInterval > 0 && m_frames > 0)
  {
    saveScreenshot(m_screenshotMode, fmt::format("screenshot_{}_{:08}.jpg", m_screenshotTimeStamp, m_frames - 1));
  }
}

void AnimatedClusters::parameterSequenceCallback(const nvutils::ParameterSequencer::State& state)
{
  std::string message;
  message += fmt::format("MemoryReport {} \"{}\" = {{ \n", state.index, state.description);
  if(m_renderer && m_scene)
  {
    Renderer::ResourceUsageInfo resource = m_renderer->getResourceUsage();

    shaderio::Readback readback;
    m_resources.getReadbackData(readback);

    const bool isRaytraceClusters = m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS;

    message += fmt::format("Memory; Bytes;\n");
    message += fmt::format("Scene; {};\n", resource.sceneMemBytes);
    if(m_tweak.renderer == RENDERER_RAYTRACE_TRIANGLES || isRaytraceClusters)
    {
      message += fmt::format("TLAS; {};\n", resource.rtTlasMemBytes);
      message += fmt::format("BLAS; {};\n", readback.blasesSize);
      if(isRaytraceClusters)
      {
        message += fmt::format("CLAS; {};\n", readback.clustersSize);
      }
      message += fmt::format("Other; {};\n", resource.rtOtherMemBytes);
    }
    message += fmt::format("Total; {};\n", resource.getTotalSum());

    message += fmt::format("Content; Count;\n");
    message += fmt::format("Clusters; {};\n", uint64_t(m_scene->m_numClusters) * m_tweak.gridCopies);
    message += fmt::format("Triangles; {};\n", uint64_t(m_scene->m_numTriangles) * m_tweak.gridCopies);
    message += fmt::format("Instances; {};\n", uint64_t(m_scene->m_instances.size()) * m_tweak.gridCopies);
  }
  message += "}\n";

  LOGI("%s", message.c_str());

  saveScreenshot(m_sequenceScreenshotMode, fmt::format("screenshot_{}_{}.jpg", state.index, state.description));
}

void AnimatedClusters::saveScreenshot(ScreenshotMode mode, const std::string& filename, int quality)
{
  if(mode == SCREENSHOT_OFF)
    return;

  // headless has no window content
  if(m_app->isHeadless())
  {
    mode = SCREENSHOT_VIEWPORT;
  }

  if(mode == SCREENSHOT_WINDOW)
  {
    m_app->saveScreenShot(std::filesystem::path(filename), quality);
  }
  else if(mode == SCREENSHOT_VIEWPORT)
  {
    // the displayed image: resolved at window size, otherwise the render size
    // the displayed image: resolved at window size, otherwise the render target (fixed resolution)
    const Resources::FrameBuffer& fb = m_resources.m_frameBuffer;
    m_app->saveImageToFile(fb.useResolved ? fb.imgColorResolved.image : fb.imgColor.image,
                           fb.useResolved ? fb.windowSize : fb.renderSize, filename, quality,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
}

void AnimatedClusters::deinitRenderer()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  if(m_renderer)
  {
    m_renderer->deinit(m_resources);
    m_renderer = nullptr;
  }
}

static VkBuildAccelerationStructureFlagsKHR getBuildFlags(AnimatedClusters::BuildMode mode)
{
  switch(mode)
  {
    case AnimatedClusters::BUILD_DEFAULT:
      return 0;
    case AnimatedClusters::BUILD_FAST_BUILD:
      return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    case AnimatedClusters::BUILD_FAST_TRACE:
      return VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    default:
      return 0;
  }
}

void AnimatedClusters::initRenderer(RendererType rtype)
{
  LOGI("Initializing renderer and compiling shaders\n");
  deinitRenderer();
  if(!m_scene)
    return;

  printf("init renderer %d\n", rtype);

  switch(rtype)
  {
    case RENDERER_RASTER_TRIANGLES:
      m_renderer = makeRendererRasterTriangles();
      break;
    case RENDERER_RASTER_CLUSTERS:
      m_renderer = makeRendererRasterClusters();
      break;
    case RENDERER_RAYTRACE_CLUSTERS:
      if(m_resources.m_supportsClusters)
      {
        m_renderer = makeRendererRayTraceClusters();
      }
      break;
    case RENDERER_RAYTRACE_TRIANGLES:
      m_renderer = makeRendererRayTraceTriangles();
      break;
  }

  m_rendererConfig.gridConfig           = m_tweak.gridConfig;
  m_rendererConfig.numSceneCopies       = m_tweak.gridCopies;
  m_rendererConfig.templateBBoxBloat    = m_tweak.templateBboxBloat;
  m_rendererConfig.positionTruncateBits = m_tweak.clusterPositionTruncationBits;
  m_rendererConfig.useTemplates         = m_tweak.useTemplates;
  m_rendererConfig.useImplicitTemplates = m_tweak.useImplicitTemplates;

  m_rendererConfig.templateBuildFlags       = getBuildFlags(m_tweak.templateBuildMode);
  m_rendererConfig.templateInstantiateFlags = getBuildFlags(m_tweak.templateInstantiateMode);
  m_rendererConfig.clusterBuildFlags        = getBuildFlags(m_tweak.clusterBuildMode);
  m_rendererConfig.clusterBlasFlags         = getBuildFlags(m_tweak.blasBuildMode);
  m_rendererConfig.triangleBuildFlags       = getBuildFlags(m_tweak.blasBuildMode);
  LOGI("build flags: template build %d, template instantiate %d, cluster build %d, blas %d, templates %d\n", m_tweak.templateBuildMode,
       m_tweak.templateInstantiateMode, m_tweak.clusterBuildMode, m_tweak.blasBuildMode, m_tweak.useTemplates);

  if(m_renderer && !m_renderer->init(m_resources, *m_scene, m_rendererConfig))
  {
    LOGE("ERROR: renderer init failied\n");
    m_renderer = nullptr;
  }

  m_rendererFboChangeID = m_resources.m_fboChangeID;
}


void AnimatedClusters::postInitNewScene()
{
  float     sceneDimension = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);
  glm::vec3 center         = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  m_frameConfig.frameConstants.wLightPos = center + sceneDimension;
  m_frameConfig.frameConstants.sceneSize = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo);

  setSceneCamera(m_sceneFilePath);

  {
    // Re-adjusting camera to fit the new scene
    m_info.cameraManipulator->fit(m_scene->m_bbox.lo, m_scene->m_bbox.hi, true);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }

  float radius = sceneDimension * 0.5f;
  m_info.cameraManipulator->setClipPlanes(glm::vec2(0.01F * radius, 100.0F * radius));
  m_info.cameraManipulator->setSpeed(radius);

  m_frameConfig.frameConstants.skyParams = {};
}

void AnimatedClusters::onAttach(nvapp::Application* app)
{
  m_app = app;

  m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_TRIANGLES, "raster triangles");
  m_ui.enumAdd(GUI_RENDERER, RENDERER_RASTER_CLUSTERS, "raster clusters");
  m_ui.enumAdd(GUI_RENDERER, RENDERER_RAYTRACE_TRIANGLES, "ray trace triangles");
  if(m_resources.m_supportsClusters)
  {
    m_ui.enumAdd(GUI_RENDERER, RENDERER_RAYTRACE_CLUSTERS, "ray trace clusters");
  }
  else
  {
    LOGW("WARNING: Cluster ray tracing extension not found\n");
    if(m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS)
    {
      m_tweak.renderer = RENDERER_RASTER_CLUSTERS;
    }
  }

  {
    for(uint32_t i = 0; i < NUM_CLUSTER_CONFIGS; i++)
    {
      std::string enumStr = fmt::format("{}T_{}V", s_clusterInfos[i].tris, s_clusterInfos[i].verts);
      m_ui.enumAdd(GUI_MESHLET, s_clusterInfos[i].cfg, enumStr.c_str());
    }
  }

  m_ui.enumAdd(GUI_SUPERSAMPLE, 1, "none");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 2, "4x");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 540, "540p");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 720, "720p");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 1080, "1080p");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 1440, "1440p");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 2160, "2160p");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 1024, "1024 sq");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 2048, "2048 sq");
  m_ui.enumAdd(GUI_SUPERSAMPLE, 4096, "4096 sq");

  m_ui.enumAdd(GUI_BUILDMODE, BUILD_DEFAULT, "default");
  m_ui.enumAdd(GUI_BUILDMODE, BUILD_FAST_BUILD, "fast build");
  m_ui.enumAdd(GUI_BUILDMODE, BUILD_FAST_TRACE, "fast trace");

  m_ui.enumAdd(GUI_TLAS_UPDATEMODE, TLAS_UPDATE_REFIT, "refit");
  m_ui.enumAdd(GUI_TLAS_UPDATEMODE, TLAS_UPDATE_REBUILD, "rebuild");

  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_NONE, "Material");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_CLUSTER, "Clusters");
  m_ui.enumAdd(GUI_VISUALIZE, VISUALIZE_TRIANGLES, "Triangles");


  // Initialize core components

  m_profilerGpuTimer.init(m_profilerTimeline, app->getDevice(), app->getPhysicalDevice(), app->getQueue(0).familyIndex, true);
  m_resources.init(app->getDevice(), app->getPhysicalDevice(), app->getInstance(), app->getQueue(0));

  {
    NVVK_CHECK(m_resources.m_samplerPool.acquireSampler(m_imguiSampler));
    NVVK_DBG_NAME(m_imguiSampler);
  }

  m_resources.initFramebuffer({128, 128}, m_tweak.supersample, m_tweak.hbaoQuarter);
  updateImguiImage();

  updatedClusterConfig();
  if(m_autoSettings)
  {
    applyAutoSettings();
  }

  // Search for default scene if none was provided on the command line
  if(m_sceneFilePath.empty())
  {
    const std::filesystem::path              exeDirectoryPath   = nvutils::getExecutablePath().parent_path();
    const std::vector<std::filesystem::path> defaultSearchPaths = {
        // regular build
        std::filesystem::absolute(exeDirectoryPath / TARGET_EXE_TO_DOWNLOAD_DIRECTORY),
        // install build
        std::filesystem::absolute(exeDirectoryPath / "resources"),
    };

    m_sceneFilePath = nvutils::findFile("bunny_v2/bunny.gltf", defaultSearchPaths);
    if(m_tweak.gridCopies == 1)
    {
      m_tweak.gridCopies = 121;  // 11x11 grid
    }
  }

  if(!m_sceneFilePath.empty())
  {
    if(initScene(m_sceneFilePath))
    {
      postInitNewScene();

      initRenderer(m_tweak.renderer);
    }
  }

  m_tweakLast          = m_tweak;
  m_sceneConfigLast    = m_sceneConfig;
  m_rendererConfigLast = m_rendererConfig;
}

void AnimatedClusters::onDetach()
{
  NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));

  deinitRenderer();
  deinitScene();

  m_resources.m_samplerPool.releaseSampler(m_imguiSampler);
  ImGui_ImplVulkan_RemoveTexture(m_imguiTexture);

  m_resources.deinit();

  m_profilerGpuTimer.deinit();
}


void AnimatedClusters::onFileDrop(const std::filesystem::path& filePath)
{
  if(filePath.empty())
    return;

  // reset grid parameter (in case scene is too large to be replicated)
  m_tweak.gridCopies = 1;
  LOGI("Loading model: %s\n", nvutils::utf8FromPath(filePath).c_str());
  deinitRenderer();

  if(initScene(filePath))
  {
    postInitNewScene();
    initRenderer(m_tweak.renderer);
    m_tweakLast       = m_tweak;
    m_sceneConfigLast = m_sceneConfig;
  }
}

void AnimatedClusters::handleChanges()
{
  if(m_tweak.clusterConfig != m_tweakLast.clusterConfig)
  {
    updatedClusterConfig();
  }

  bool sceneChanged = false;
  if(memcmp(&m_sceneConfig, &m_sceneConfigLast, sizeof(m_sceneConfig)))
  {
    sceneChanged = true;

    deinitRenderer();
    initScene(m_sceneFilePath);
  }

  bool shaderChanged = false;
  if(m_reloadShaders)
  {
    shaderChanged   = true;
    m_reloadShaders = false;
  }

  bool frameBufferChanged = false;
  // a change of only the HBAO resolution leaves the change id alone
  const bool supersampleChanged = tweakChanged(m_tweak.supersample);
  const bool hbaoQuarterChanged = tweakChanged(m_tweak.hbaoQuarter);
  if(supersampleChanged || hbaoQuarterChanged)
  {
    const size_t fboChangeID = m_resources.m_fboChangeID;
    m_resources.initFramebuffer(m_windowSize, m_tweak.supersample, m_tweak.hbaoQuarter);
    if(fboChangeID != m_resources.m_fboChangeID)
    {
      updateImguiImage();
      frameBufferChanged = true;
    }
  }

  if(m_autoSettings && rendererCfgChanged(m_rendererConfig.doAnimation))
  {
    applyAutoSettings();
  }

  bool rendererChanged = false;
  if(sceneChanged || shaderChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.gridCopies)
     || tweakChanged(m_tweak.gridConfig) || rendererCfgChanged(m_rendererConfig.doAnimation)
     || (m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS
         && (tweakChanged(m_tweak.clusterPositionTruncationBits) || tweakChanged(m_tweak.templateBboxBloat)
             || tweakChanged(m_tweak.templateBuildMode) || tweakChanged(m_tweak.clusterBuildMode)
             || tweakChanged(m_tweak.useTemplates) || tweakChanged(m_tweak.useImplicitTemplates)
             || tweakChanged(m_tweak.templateInstantiateMode) || tweakChanged(m_tweak.blasBuildMode)))
     || (m_tweak.renderer == RENDERER_RAYTRACE_TRIANGLES && (tweakChanged(m_tweak.blasBuildMode))))
  {
    rendererChanged = true;

    initRenderer(m_tweak.renderer);
  }
  else if(m_renderer && frameBufferChanged)
  {
    m_renderer->updatedFrameBuffer(m_resources);
  }

  bool hadChange = memcmp(&m_tweakLast, &m_tweak, sizeof(m_tweak))
                   || memcmp(&m_rendererConfigLast, &m_rendererConfig, sizeof(m_rendererConfig))
                   || memcmp(&m_sceneConfigLast, &m_sceneConfig, sizeof(m_sceneConfig));

  m_tweakLast          = m_tweak;
  m_rendererConfigLast = m_rendererConfig;
  m_sceneConfigLast    = m_sceneConfig;

  if(hadChange && m_tweak.autoResetTimers)
  {
    m_info.profilerManager->resetFrameSections(8);
  }
}

void AnimatedClusters::applyAutoSettings()
{
  const BuildMode mode = m_rendererConfig.doAnimation ? BuildMode::BUILD_FAST_BUILD : BuildMode::BUILD_FAST_TRACE;

  m_tweak.blasBuildMode           = mode;
  m_tweak.templateInstantiateMode = mode;
  m_tweak.clusterBuildMode        = mode;
  m_tweak.useTemplates            = m_rendererConfig.doAnimation;
}

void AnimatedClusters::registerCameraPathParameters()
{
  // `--addcamerapath "<string>"` appends a path definition; the index is the
  // order in which the paths were added. Repeatable.
  m_info.parameterRegistry->addCustom(
      {"addcamerapath", "append a camera path definition string (repeatable); the first one replaces the scene's paths, the path index equals the order added"},
      1, [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path&) -> bool {
        CameraPath path;
        if(!path.setFromString(args[0]))
        {
          LOGE("addcamerapath: could not parse camera path string\n");
          return false;
        }
        if(!m_cameraPathsExternal)
        {
          // the first explicit path replaces the ones loaded from the scene's file
          m_cameraPaths.clear();
          m_cameraPathActive   = -1;
          m_cameraPathEditKey  = -1;
          m_cameraPathPlayback = CAMERA_PATH_STOP;
        }
        m_cameraPaths.push_back(std::move(path));
        m_cameraPathsExternal = true;
        if(m_cameraPathActive < 0)
          m_cameraPathActive = 0;
        LOGI("addcamerapath: added path %d with %zu keyframes\n", int(m_cameraPaths.size()) - 1, m_cameraPaths.back().size());
        return true;
      });

  // `--loadcamerapaths "<filename>"` replaces all path definitions with the
  // paths from a simple text file (one path per line). Full overwrite.
  m_info.parameterRegistry->addCustom(
      {"loadcamerapaths", "load all camera path definitions from a text file (full overwrite)"}, 1,
      [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path& basePath) -> bool {
        std::filesystem::path file = std::filesystem::path(args[0]);
        if(file.is_relative() && !basePath.empty())
          file = basePath / file;
        if(!loadCameraPaths(file, m_cameraPaths))
        {
          LOGE("loadcamerapaths: could not open %s\n", nvutils::utf8FromPath(file).c_str());
          return false;
        }
        m_cameraPathActive    = m_cameraPaths.empty() ? -1 : 0;
        m_cameraPathPlayback  = CAMERA_PATH_STOP;
        m_cameraPathsExternal = true;
        LOGI("loadcamerapaths: loaded %zu path(s) from %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(file).c_str());
        return true;
      });

  // `--runcamerapath <index> <framecount>` plays back a path over a fixed number
  // of frames (deterministic), one keyframe-step per rendered frame. Meant to be
  // used within benchmark sequences.
  m_info.parameterRegistry->addCustom(
      {"runcamerapath", "deterministic fixed-step playback: <index> <framecount>, spreads the given path across framecount frames"},
      2, [this](const nvutils::ParameterBase* const, std::span<char const* const> args, const std::filesystem::path&) -> bool {
        int index      = atoi(args[0]);
        int frameCount = atoi(args[1]);
        if(index < 0 || index >= int(m_cameraPaths.size()))
        {
          LOGE("runcamerapath: path index %d out of range (%d paths defined)\n", index, int(m_cameraPaths.size()));
          return false;
        }
        m_cameraPathActive   = index;
        m_cameraPathFrames   = std::max(frameCount, 1);
        m_cameraPathFrame    = 0;
        m_cameraPathPlayback = CAMERA_PATH_FIXED;
        return true;
      });
}

std::filesystem::path AnimatedClusters::sceneCameraPathsFile(const std::filesystem::path& scenePath) const
{
  if(scenePath.empty())
    return {};

  // stored next to the model file, named after it, e.g. ".../bunny.camerapaths.txt"
  std::filesystem::path file = scenePath;
  file.replace_extension(".camerapaths.txt");
  return file;
}

bool AnimatedClusters::applyCameraPathSample(double u)
{
  if(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()))
    return false;

  CameraPathKey key;
  if(!m_cameraPaths[m_cameraPathActive].sample(u, key))
    return false;

  // keep the manipulator's other camera state (clip planes, projection, ...)
  nvutils::CameraManipulator::Camera cam = m_info.cameraManipulator->getCamera();
  cam.eye                                = key.eye;
  cam.ctr                                = key.ctr;
  cam.up                                 = key.up;
  cam.fov                                = key.fov;
  m_info.cameraManipulator->setCamera(cam, true);
  return true;
}

void AnimatedClusters::updateCameraPath(double time)
{
  if(m_cameraPathPlayback == CAMERA_PATH_STOP)
    return;

  if(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()) || m_cameraPaths[m_cameraPathActive].empty())
  {
    m_cameraPathPlayback = CAMERA_PATH_STOP;
    return;
  }

  const CameraPath& path = m_cameraPaths[m_cameraPathActive];

  if(m_cameraPathPlayback == CAMERA_PATH_FIXED)
  {
    // deterministic: frame f maps to u = f / (frames - 1)
    int    denom = std::max(m_cameraPathFrames - 1, 1);
    double u     = double(std::min(m_cameraPathFrame, denom)) / double(denom);
    applyCameraPathSample(u);
    m_cameraPathU = u;

    // advance for the next frame, hold at the end (the sequence/headless run
    // stops on its own frame count)
    if(m_cameraPathFrame < m_cameraPathFrames - 1)
      m_cameraPathFrame++;
  }
  else  // CAMERA_PATH_REALTIME
  {
    double dt           = m_cameraPathStarted ? std::max(time - m_lastTime, 0.0) : 0.0;
    m_cameraPathStarted = true;
    if(path.duration > 0.0)
      m_cameraPathU += dt / path.duration;

    if(m_cameraPathU >= 1.0)
    {
      if(path.loop)
        m_cameraPathU = std::fmod(m_cameraPathU, 1.0);
      else
      {
        // this frame still shows the last key
        m_cameraPathU        = 1.0;
        m_cameraPathPlayback = CAMERA_PATH_STOP;
      }
    }

    applyCameraPathSample(m_cameraPathU);
  }
}

void AnimatedClusters::onRender(VkCommandBuffer cmd)
{
  // headless runs step the animation per frame, which keeps screenshots repeatable
  double timeStep = m_fixedTimeStep >= 0 ? double(m_fixedTimeStep) : (m_app->isHeadless() ? 1.0 / 60.0 : 0.0);
  double time     = timeStep > 0 ? double(m_frames) * timeStep : m_clock.getSeconds();

  // drive the camera along the active path (overrides the manipulator for the
  // rendered frame); runs after nvapp::ElementCamera's per-frame input update
  updateCameraPath(time);

  m_resources.beginFrame(m_app->getFrameCycleIndex());

  m_frameConfig.windowSize = m_windowSize;
  m_frameConfig.hbaoActive = false;

  if(m_renderer)
  {
    uint32_t width  = m_windowSize.width;
    uint32_t height = m_windowSize.height;

    if(m_rendererFboChangeID != m_resources.m_fboChangeID)
    {
      m_renderer->updatedFrameBuffer(m_resources);
      m_rendererFboChangeID = m_resources.m_fboChangeID;
    }

    shaderio::FrameConstants& frameConstants = m_frameConfig.frameConstants;

    uint32_t renderWidth  = m_resources.m_frameBuffer.renderSize.width;
    uint32_t renderHeight = m_resources.m_frameBuffer.renderSize.height;
    frameConstants.time   = float(time);

    // wireframe overlay disabled for all clusters
    frameConstants.visFilterClusterID  = ~0;
    frameConstants.visFilterInstanceID = ~0;

    frameConstants.facetShading = m_tweak.facetShading ? 1 : 0;
    frameConstants.doAnimation  = m_rendererConfig.doAnimation ? 1 : 0;

    m_frameConfig.hbaoActive =
        m_tweak.hbaoActive && (m_tweak.renderer == RENDERER_RASTER_CLUSTERS || m_tweak.renderer == RENDERER_RASTER_TRIANGLES);

    if(m_rendererConfig.doAnimation)
    {
      if(m_tweak.overrideTime)
      {
        m_frameConfig.frameConstants.animationState = m_tweak.overrideTime;
        m_animTime                                  = m_tweak.overrideTime;
      }
      else
      {
        m_animTime += (time - m_lastTime) * 0.5;
        m_frameConfig.frameConstants.animationState = float(m_animTime);
      }
    }
    frameConstants.bgColor = m_resources.m_bgColor;

    frameConstants.viewport    = glm::ivec2(renderWidth, renderHeight);
    frameConstants.viewportf   = glm::vec2(renderWidth, renderHeight);
    frameConstants.supersample = std::max(int(std::lround(m_resources.m_frameBuffer.pixelScale)), 1);
    frameConstants.nearPlane   = float(m_info.cameraManipulator->getClipPlanes().x);
    frameConstants.farPlane    = float(m_info.cameraManipulator->getClipPlanes().y);
    frameConstants.wUpDir      = glm::vec3(m_info.cameraManipulator->getUp());

    // reversed-Z: near plane at depth 1, far plane at 0
    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(float(m_info.cameraManipulator->getFov())),
                                                 float(renderWidth) / float(renderHeight), frameConstants.farPlane,
                                                 frameConstants.nearPlane);
    projection[1][1] *= -1;

    glm::mat4 view  = m_info.cameraManipulator->getViewMatrix();
    glm::mat4 viewI = glm::inverse(view);

    frameConstants.viewProjMatrix  = projection * view;
    frameConstants.viewProjMatrixI = glm::inverse(frameConstants.viewProjMatrix);
    frameConstants.viewMatrix      = view;
    frameConstants.viewMatrixI     = viewI;
    frameConstants.projMatrix      = projection;
    frameConstants.projMatrixI     = glm::inverse(projection);

    glm::mat4 viewNoTrans         = view;
    viewNoTrans[3]                = {0.0f, 0.0f, 0.0f, 1.0f};
    frameConstants.skyProjMatrixI = glm::inverse(projection * viewNoTrans);

    glm::vec4 hPos   = projection * glm::vec4(1.0f, 1.0f, -frameConstants.farPlane, 1.0f);
    glm::vec2 hCoord = glm::vec2(hPos.x / hPos.w, hPos.y / hPos.w);
    glm::vec2 dim    = glm::abs(hCoord);

    // helper to quickly get footprint of a point at a given distance
    //
    // __.__hPos (far plane is width x height)
    // \ | /
    //  \|/
    //   x camera
    //
    // here: viewPixelSize / point.w = size of point in pixels
    // * 0.5f because renderWidth/renderHeight represents [-1,1] but we need half of frustum
    frameConstants.viewPixelSize = dim * (glm::vec2(float(renderWidth), float(renderHeight)) * 0.5f) * frameConstants.farPlane;
    // here: viewClipSize / point.w = size of point in clip-space units
    // no extra scale as half clip space is 1.0 in extent
    frameConstants.viewClipSize = dim * frameConstants.farPlane;

    frameConstants.viewPos = frameConstants.viewMatrixI[3];  // position of eye in the world
    frameConstants.viewDir = -viewI[2];

    frameConstants.viewPlane   = frameConstants.viewDir;
    frameConstants.viewPlane.w = -glm::dot(glm::vec3(frameConstants.viewPos), glm::vec3(frameConstants.viewDir));

    frameConstants.wLightPos = frameConstants.viewMatrixI[3];  // place light at position of eye in the world

    {
      // hbao setup
      auto& hbaoView            = m_frameConfig.hbaoSettings.view;
      hbaoView.farPlane         = frameConstants.farPlane;
      hbaoView.nearPlane        = frameConstants.nearPlane;
      hbaoView.projectionMatrix = projection;
      hbaoView.viewMatrix       = view;
      hbaoView.tanFovy          = tanf(glm::radians(float(m_info.cameraManipulator->getFov())) * 0.5f);

      m_frameConfig.hbaoSettings.radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * m_tweak.hbaoRadius;
    }

    m_renderer->render(cmd, m_resources, *m_scene, m_frameConfig, m_profilerGpuTimer);
  }
  else
  {
    m_resources.emptyFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  {
    m_resources.blitFrame(cmd, m_frameConfig, m_profilerGpuTimer);
  }

  m_resources.endFrame();

  m_lastTime = time;
  m_frames++;
}

void AnimatedClusters::setSceneCamera(const std::filesystem::path& filePath)
{
  nvgui::SetCameraJsonFile(filePath);

  float     radius = glm::length(m_scene->m_bbox.hi - m_scene->m_bbox.lo) * 0.5f;
  glm::vec3 center = (m_scene->m_bbox.hi + m_scene->m_bbox.lo) * 0.5f;

  if(!m_scene->m_cameras.empty())
  {
    auto& c = m_scene->m_cameras[0];
    m_info.cameraManipulator->setFov(c.fovy);


    c.eye              = glm::vec3(c.worldMatrix[3]);
    float     distance = glm::length(center - c.eye);
    glm::mat3 rotMat   = glm::mat3(c.worldMatrix);
    c.center           = {0, 0, -distance};
    c.center           = c.eye + (rotMat * c.center);
    c.up               = {0, 1, 0};

    m_info.cameraManipulator->setCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});

    nvgui::SetHomeCamera({c.eye, c.center, c.up, static_cast<float>(glm::degrees(c.fovy))});
    for(auto& cam : m_scene->m_cameras)
    {
      cam.eye            = glm::vec3(cam.worldMatrix[3]);
      float     distance = glm::length(center - cam.eye);
      glm::mat3 rotMat   = glm::mat3(cam.worldMatrix);
      cam.center         = {0, 0, -distance};
      cam.center         = cam.eye + (rotMat * cam.center);
      cam.up             = {0, 1, 0};


      nvgui::AddCamera({cam.eye, cam.center, cam.up, static_cast<float>(glm::degrees(cam.fovy))});
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    m_info.cameraManipulator->fit(m_scene->m_bbox.lo, m_scene->m_bbox.hi, true);
    nvgui::SetHomeCamera(m_info.cameraManipulator->getCamera());
  }
}

float AnimatedClusters::decodePickingDepth(const shaderio::Readback& readback)
{
  if(!isPickingValid(readback))
  {
    return 0.f;
  }
  uint32_t bits = readback._packedDepth0;
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  float res = *(float*)&bits;
  // reversed-Z depth as written by the shaders (1 = near plane, 0 = far plane)
  return res;
}

bool AnimatedClusters::isPickingValid(const shaderio::Readback& readback)
{
  return readback._packedDepth0 != 0u;
}

}  // namespace animatedclusters
