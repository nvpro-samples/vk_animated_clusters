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

#include <memory>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/compute_pipeline.hpp>

#include "resources.hpp"
#include "scene.hpp"

namespace animatedclusters {
struct RendererConfig
{
  // scene related
  uint32_t  numSceneCopies = 1;
  uint32_t  gridConfig     = 3;
  glm::vec3 refShift       = glm::vec3(1, 1, 1);

  bool doAnimation = true;

  // rt related
  VkBuildAccelerationStructureFlagsKHR triangleBuildFlags = 0;

  // cluster related
  VkBuildAccelerationStructureFlagsKHR clusterBlasFlags         = 0;
  VkBuildAccelerationStructureFlagsKHR clusterBuildFlags        = 0;
  VkBuildAccelerationStructureFlagsKHR templateBuildFlags       = 0;
  VkBuildAccelerationStructureFlagsKHR templateInstantiateFlags = 0;
  bool                                 useTemplates             = true;
  bool                                 useImplicitTemplates     = true;
  uint32_t                             positionTruncateBits     = 0;
  float                                templateBBoxBloat        = 0.1f;
};

class Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) = 0;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) = 0;
  virtual void deinit(Resources& res) = 0;
  virtual ~Renderer() {};  // Defined only so that inherited classes also have virtual destructors. Use deinit().
  virtual void updatedFrameBuffer(Resources& res) {};

  struct ResourceUsageInfo
  {
    size_t rtBlasMemBytes{};
    size_t rtTlasMemBytes{};
    size_t rtClasMemBytes{};
    size_t rtOtherMemBytes{};
    size_t sceneMemBytes{};

    size_t getTotalSum() const
    {
      return rtBlasMemBytes + rtTlasMemBytes + rtClasMemBytes + rtOtherMemBytes + sceneMemBytes;
    }
  };

  inline ResourceUsageInfo getResourceUsage() const { return m_resourceUsageInfo; };

protected:
  bool initBasicShaders(Resources& res, Scene& scene, const RendererConfig& config);
  void initBasics(Resources& res, Scene& scene, const RendererConfig& config);
  void deinitBasics(Resources& res);

  void updateAnimation(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);

  void initRayTracingTlas(Resources& res, Scene& scene, const RendererConfig& config, const VkAccelerationStructureKHR* blas = nullptr);
  void updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, Scene& scene, bool update = false);

  RendererConfig m_config;

  struct BasicShaders
  {
    shaderc::SpvCompilationResult animComputeVertices;
    shaderc::SpvCompilationResult animComputeNormals;
  } m_basicShaders;

  struct BasicPipelines
  {
    VkPipeline animComputeNormals{};
    VkPipeline animComputeVertices{};
  } m_basicPipelines;

  VkPipelineLayout m_animPipelineLayout{};

  struct RenderInstanceData
  {
    nvvk::Buffer positions;
    nvvk::Buffer normals;
  };

  std::vector<uint32_t> m_geometryFirstInstance;

  std::vector<shaderio::RenderInstance> m_renderInstances;
  nvvk::Buffer                          m_renderInstanceBuffer;
  std::vector<RenderInstanceData>       m_renderInstanceBuffers;

  nvvk::Buffer                                m_tlasInstancesBuffer;
  VkAccelerationStructureGeometryKHR          m_tlasGeometry{};
  VkAccelerationStructureBuildGeometryInfoKHR m_tlasBuildInfo{};
  nvvk::Buffer                                m_tlasScratchBuffer;
  nvvk::AccelerationStructure                 m_tlas;

  ResourceUsageInfo m_resourceUsageInfo{};
};

std::unique_ptr<Renderer> makeRendererRasterTriangles();
std::unique_ptr<Renderer> makeRendererRasterClusters();
std::unique_ptr<Renderer> makeRendererRayTraceTriangles();
std::unique_ptr<Renderer> makeRendererRayTraceClusters();
}  // namespace animatedclusters
