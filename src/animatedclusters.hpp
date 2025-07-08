/*
* Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
* SPDX-License-Identifier: Apache-2.0
*/

#pragma once

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvvk/context.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvgui/enum_registry.hpp>

#include "renderer.hpp"


namespace animatedclusters {

class AnimatedClusters : public nvapp::IAppElement
{
public:
  enum RendererType
  {
    RENDERER_RASTER_TRIANGLES,
    RENDERER_RASTER_CLUSTERS,
    RENDERER_RAYTRACE_TRIANGLES,
    RENDERER_RAYTRACE_CLUSTERS,
  };

  enum ClusterConfig
  {
    CLUSTER_64T_64V,
    CLUSTER_64T_128V,
    CLUSTER_64T_192V,
    CLUSTER_96T_96V,
    CLUSTER_128T_128V,
    CLUSTER_128T_256V,
    CLUSTER_256T_256V,
    NUM_CLUSTER_CONFIGS,
  };

  struct ClusterInfo
  {
    uint32_t      tris;
    uint32_t      verts;
    ClusterConfig cfg;
  };

  static const ClusterInfo s_clusterInfos[NUM_CLUSTER_CONFIGS];

  enum BuildMode
  {
    BUILD_DEFAULT,
    BUILD_FAST_BUILD,
    BUILD_FAST_TRACE,
  };

  enum TLASUpdateMode
  {
    TLAS_UPDATE_REFIT,
    TLAS_UPDATE_REBUILD
  };

  enum GuiEnums
  {
    GUI_RENDERER,
    GUI_SUPERSAMPLE,
    GUI_MESHLET,
    GUI_BUILDMODE,
    GUI_TLAS_UPDATEMODE,
    GUI_VISUALIZE,
  };

  struct Tweak
  {
    ClusterConfig clusterConfig = CLUSTER_64T_64V;

    RendererType renderer    = RENDERER_RAYTRACE_CLUSTERS;
    int          supersample = 2;

    bool  hbaoFullRes = false;
    bool  hbaoActive  = true;
    float hbaoRadius  = 0.05f;

    float     overrideTime                  = 0.0f;
    bool      facetShading                  = true;
    bool      useTemplates                  = true;
    bool      useImplicitTemplates          = false;
    float     templateBboxBloat             = 0.5f;
    BuildMode templateInstantiateMode       = BuildMode::BUILD_FAST_BUILD;
    BuildMode templateBuildMode             = BuildMode::BUILD_FAST_TRACE;
    BuildMode clusterBuildMode              = BuildMode::BUILD_FAST_TRACE;
    BuildMode blasBuildMode                 = BuildMode::BUILD_FAST_BUILD;
    uint32_t  clusterPositionTruncationBits = 0;

    uint32_t gridCopies = 1;
    uint32_t gridConfig = 13;

    bool autoResetTimers = false;
  };

  struct ViewPoint
  {
    std::string name;
    glm::mat4   mat;
    float       sceneScale;
    float       fov;
  };

  struct TargetImage
  {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
  };

  struct Info
  {
    nvutils::ProfilerManager*                   profilerManager{};
    nvutils::ParameterRegistry*                 parameterRegistry{};
    std::shared_ptr<nvutils::CameraManipulator> cameraManipulator;
  };

  AnimatedClusters(const Info& info);

  ~AnimatedClusters() override { m_info.profilerManager->destroyTimeline(m_profilerTimeline); }

  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onUIRender() override;
  void onUIMenu() override;
  void onPreRender() override;
  void onRender(VkCommandBuffer cmd) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onFileDrop(const std::filesystem::path& filename) override;

  void setSupportsClusters(bool supported) { m_resources.m_supportsClusters = supported; }

private:
  VkExtent2D                 m_windowSize;
  Info                       m_info;
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer{};
  nvapp::Application*        m_app{};

  //////////////////////////////////////////////////////////////////////////

  // key components

  Resources                 m_resources;
  FrameConfig               m_frameConfig;
  double                    m_lastTime = 0;
  VkDescriptorSet           m_imguiTexture{};
  VkSampler                 m_imguiSampler{};
  nvgui::EnumRegistry       m_ui;
  nvutils::PerformanceTimer m_clock;

  bool   m_reloadShaders         = false;
  bool   m_requestCameraRecenter = false;
  int    m_frames                = 0;
  double m_animTime              = 0;

  Tweak m_tweak;
  Tweak m_tweakLast;

  std::unique_ptr<Scene> m_scene;
  std::filesystem::path  m_sceneFilePath;
  SceneConfig            m_sceneConfig;
  SceneConfig            m_sceneConfigLast;
  glm::vec3              m_sceneUpVector = glm::vec3(0, 1, 0);

  std::unique_ptr<Renderer> m_renderer;
  uint64_t                  m_rendererFboChangeID{};
  RendererConfig            m_rendererConfig;
  RendererConfig            m_rendererConfigLast;

  bool initScene(const std::filesystem::path& filePath);
  void setSceneCamera(const std::filesystem::path& filePath);
  void deinitScene();
  void postInitNewScene();

  void initRenderer(RendererType rtype);
  void deinitRenderer();

  void updateImguiImage();

  void adjustSceneClusterConfig();
  void updatedClusterConfig();
  void handleChanges();

  float decodePickingDepth(const shaderio::Readback& readback);
  bool  isPickingValid(const shaderio::Readback& readback);

  void viewportUI(ImVec2 corner);

  template <typename T>
  bool sceneChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_sceneConfig);
    assert(offset < sizeof(m_sceneConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_sceneConfigLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool tweakChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    assert(offset < sizeof(m_tweak));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_tweakLast) + offset, sizeof(T)) != 0;
  }

  template <typename T>
  bool rendererCfgChanged(const T& val) const
  {
    size_t offset = size_t(&val) - size_t(&m_rendererConfig);
    assert(offset < sizeof(m_rendererConfig));
    return memcmp(&val, reinterpret_cast<const uint8_t*>(&m_rendererConfigLast) + offset, sizeof(T)) != 0;
  }
};
}  // namespace animatedclusters
