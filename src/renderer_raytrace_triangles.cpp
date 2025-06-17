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

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/sbtwrapper_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvkhl/pipeline_container.hpp>
#include <nvh/alignment.hpp>

#include "renderer.hpp"

namespace animatedclusters {

class RendererRayTraceTriangles : public Renderer
{
private:
  static constexpr VkDeviceSize MAX_BLAS_SCRATCH_BUFFER_SIZE = VkDeviceSize(2) * 1024 * 1024 * 1024;

public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res) override;

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  void initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config);
  void updateRayTracingBlas(VkCommandBuffer cmd, Resources& res, Scene& scene, float rebuildFraction = 1.f);

  // without animation allow compaction
  void queryCompactRayTracingBlas(VkCommandBuffer cmd, VkQueryPool queryPool, Resources& res, Scene& scene);
  void compactRayTracingBlas(VkCommandBuffer cmd, VkQueryPool queryPool, std::vector<nvvk::AccelKHR>& oldBlas, Resources& res, Scene& scene);

  void initRayTracingPipeline(Resources& res);

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  
  nvvk::SBTWrapper          m_rtSbt;   // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout

  struct Shaders
  {
    nvvk::ShaderModuleID rayGenShader;
    nvvk::ShaderModuleID closestHitShader;
    nvvk::ShaderModuleID missShader;
    nvvk::ShaderModuleID missShaderAO;
  } m_shaders;

  nvvk::DescriptorSetContainer m_dsetContainer;

  // auxiliary rendering data

  std::vector<VkAccelerationStructureGeometryKHR>          m_blasGeometries;
  std::vector<VkAccelerationStructureBuildGeometryInfoKHR> m_blasBuildInfos;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR>    m_blasRangeInfos;
  std::vector<VkAccelerationStructureBuildSizesInfoKHR>    m_blasSizeInfos;

  RBuffer                     m_blasScratchBuffer;
  std::vector<nvvk::AccelKHR> m_blas;

  VkDeviceSize m_compactSize{0};

  uint32_t m_currentBlasRebuildInstanceStart{0u};
};

bool RendererRayTraceTriangles::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  // do shaders first, most likely to contain errors
  m_shaders.rayGenShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl");
  m_shaders.closestHitShader =
      res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "render_raytrace_triangles.rchit.glsl");
  m_shaders.missShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                                "#define RAYTRACING_PAYLOAD_INDEX 0\n");
  m_shaders.missShaderAO = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                                  "#define RAYTRACING_PAYLOAD_INDEX 1\n");

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRayTraceTriangles::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_config = config;

  if(!initShaders(res, scene, config))
    return false;

  initBasics(res, scene, config);

  m_resourceUsageInfo.sceneMemBytes += scene.m_sceneTriangleMemBytes;

  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProperties};
  m_rtProperties.pNext = &m_accProperties;
  vkGetPhysicalDeviceProperties2(res.m_physical, &prop2);

  initRayTracingBlas(res, scene, config);

  if(m_config.doAnimation)
  {
    std::vector<VkAccelerationStructureKHR> blas(m_blas.size());
    for(size_t i = 0; i < blas.size(); i++)
    {
      blas[i] = m_blas[i].accel;
    }

    initRayTracingTlas(res, scene, config, blas.data());

    VkCommandBuffer cmd = res.createTempCmdBuffer();

    updateRayTracingBlas(cmd, res, scene);
    updateRayTracingTlas(cmd, res, scene);

    res.tempSyncSubmit(cmd);
  }
  else
  {
    uint32_t blasCount = static_cast<uint32_t>(m_blas.size());

    // for compaction we need a querypool to get the compacted blas sizes
    VkQueryPoolCreateInfo poolCreateInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    poolCreateInfo.queryCount            = blasCount;
    poolCreateInfo.queryType             = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    VkQueryPool queryPool                = nullptr;
    vkCreateQueryPool(res.m_device, &poolCreateInfo, nullptr, &queryPool);
    vkResetQueryPool(res.m_device, queryPool, 0, blasCount);

    VkCommandBuffer cmd = res.createTempCmdBuffer();

    updateRayTracingBlas(cmd, res, scene);
    queryCompactRayTracingBlas(cmd, queryPool, res, scene);

    res.tempSyncSubmit(cmd);

    cmd = res.createTempCmdBuffer();

    std::vector<nvvk::AccelKHR> oldBlas;
    compactRayTracingBlas(cmd, queryPool, oldBlas, res, scene);

    // feed most recent m_blas into tlas
    std::vector<VkAccelerationStructureKHR> blas(m_renderInstances.size());
    for(size_t i = 0; i < m_renderInstances.size(); i++)
    {
      blas[i] = m_blas[m_renderInstances[i].geometryID].accel;
    }
    initRayTracingTlas(res, scene, config, blas.data());
    // update tlas along
    updateRayTracingTlas(cmd, res, scene);

    res.tempSyncSubmit(cmd);

    // no longer need pool
    vkDestroyQueryPool(res.m_device, queryPool, nullptr);
    // nor the old blas, after compaction we can delete them
    for(uint32_t idx = 0; idx < blasCount; idx++)
    {
      res.destroy(oldBlas[idx]);
    }
  }


  initRayTracingPipeline(res);
  return true;
}

void RendererRayTraceTriangles::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  if(m_config.doAnimation)
  {
    updateAnimation(primary, res, scene, frame, profiler);
    {
      auto timerSection = profiler.timeRecurring("AS Build/Refit", primary);
      {
        auto timerSection2 = profiler.timeRecurring("blas", primary);
        updateRayTracingBlas(primary, res, scene, frame.blasRebuildFraction);
      }
      {
        auto timerSection2 = profiler.timeRecurring("tlas", primary);
        updateRayTracingTlas(primary, res, scene, !frame.forceTlasFullRebuild);
      }
    }
  }

  shaderio::Readback readback = {};
  readback.blasesSize         = m_config.doAnimation ? m_resourceUsageInfo.rtBlasMemBytes : m_compactSize;

  vkCmdUpdateBuffer(primary, res.m_common.view.buffer, 0, sizeof(shaderio::FrameConstants), (const uint32_t*)&frame.frameConstants);
  vkCmdUpdateBuffer(primary, res.m_common.readbackDevice.buffer, 0, sizeof(shaderio::Readback), &readback);

  VkMemoryBarrier memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  res.cmdImageTransition(primary, res.m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

  // Ray trace
  {
    auto timerSection = profiler.timeRecurring("Render", primary);

    if(frame.drawObjects)
    {
      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
      vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, 1,
                              m_dsetContainer.getSets(), 0, nullptr);

      const std::array<VkStridedDeviceAddressRegionKHR, 4>& bindingTables = m_rtSbt.getRegions();
      vkCmdTraceRaysKHR(primary, &bindingTables[0], &bindingTables[1], &bindingTables[2], &bindingTables[3],
                        frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);
    }
  }
}

void RendererRayTraceTriangles::initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config)
{
  uint32_t blasCount = static_cast<uint32_t>(config.doAnimation ? m_renderInstances.size() : scene.m_geometries.size());
  VkDeviceSize totalBlasSize{0};     // Memory size of all allocated BLAS
  VkDeviceSize totalScratchSize{0};  // Total ScratchSize
  VkDeviceSize maxScratchSize{0};    // Total ScratchSize

  m_blasGeometries.resize(blasCount);
  m_blasBuildInfos.resize(blasCount);
  m_blasRangeInfos.resize(blasCount);
  m_blasSizeInfos.resize(blasCount);

  m_blas.resize(blasCount);

  for(uint32_t idx = 0; idx < blasCount; idx++)
  {
    uint32_t instanceIdx = config.doAnimation ? idx : m_geometryFirstInstance[idx];

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};

    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = m_renderInstances[instanceIdx].positions;
    triangles.vertexStride             = sizeof(glm::vec3);
    triangles.maxVertex                = m_renderInstances[instanceIdx].numVertices - 1;
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = m_renderInstances[instanceIdx].triangles;


    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.geometry.triangles = triangles;
    geom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR;

    m_blasGeometries[idx] = geom;

    VkAccelerationStructureBuildRangeInfoKHR offset{};

    offset.primitiveCount  = m_renderInstances[instanceIdx].numTriangles;
    offset.primitiveOffset = 0;
    offset.firstVertex     = 0;
    offset.transformOffset = 0;


    m_blasRangeInfos[idx] = offset;

    // Filling partially the VkAccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
    // Other information will be filled in the createBlas (see #2)
    m_blasBuildInfos[idx].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    m_blasBuildInfos[idx].type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    m_blasBuildInfos[idx].mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    m_blasBuildInfos[idx].flags = (m_config.doAnimation ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR :
                                                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
                                  | config.triangleBuildFlags;
    m_blasBuildInfos[idx].geometryCount = 1;
    m_blasBuildInfos[idx].pGeometries   = &m_blasGeometries[idx];

    m_blasSizeInfos[idx].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    // Finding sizes to create acceleration structures and scratch
    vkGetAccelerationStructureBuildSizesKHR(res.m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &m_blasBuildInfos[idx], &offset.primitiveCount, &m_blasSizeInfos[idx]);

    // Extra info
    totalBlasSize += m_blasSizeInfos[idx].accelerationStructureSize;
    totalScratchSize +=
        nvh::align_up(m_blasSizeInfos[idx].buildScratchSize, m_accProperties.minAccelerationStructureScratchOffsetAlignment);
    maxScratchSize = std::max(maxScratchSize, m_blasSizeInfos[idx].buildScratchSize);
  }

  // Allocate the scratch buffers holding the temporary data of the acceleration structure builder

  m_blasScratchBuffer = res.createBuffer(std::min(std::max(maxScratchSize, MAX_BLAS_SCRATCH_BUFFER_SIZE), totalScratchSize),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_blasScratchBuffer.info.range;

  for(uint32_t idx = 0; idx < blasCount; idx++)
  {
    // Actual allocation of buffer and acceleration structure.
    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    createInfo.size = m_blasSizeInfos[idx].accelerationStructureSize;  // Will be used to allocate memory.
    m_blas[idx]     = res.createAccelKHR(createInfo);
    m_resourceUsageInfo.rtBlasMemBytes += createInfo.size;

    // BuildInfo #2 part
    m_blasBuildInfos[idx].dstAccelerationStructure = m_blas[idx].accel;  // Setting where the build lands
  }
}

void RendererRayTraceTriangles::queryCompactRayTracingBlas(VkCommandBuffer cmd, VkQueryPool queryPool, Resources& res, Scene& scene)
{
  assert(!m_config.doAnimation);

  uint32_t blasCount = static_cast<uint32_t>(m_blas.size());

  std::vector<VkAccelerationStructureKHR> structures(blasCount);
  for(uint32_t idx = 0; idx < blasCount; idx++)
  {
    structures[idx] = m_blas[idx].accel;
  }

  vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, blasCount, structures.data(),
                                                VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, 0);
}

void RendererRayTraceTriangles::compactRayTracingBlas(VkCommandBuffer              cmd,
                                                      VkQueryPool                  queryPool,
                                                      std::vector<nvvk::AccelKHR>& oldBlas,
                                                      Resources&                   res,
                                                      Scene&                       scene)
{
  assert(!m_config.doAnimation);

  uint32_t blasCount = static_cast<uint32_t>(m_blas.size());

  oldBlas.resize(blasCount);

  std::vector<uint64_t> compactSizes(blasCount);

  vkGetQueryPoolResults(res.m_device, queryPool, 0, blasCount, sizeof(uint64_t) * blasCount, compactSizes.data(),
                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  for(uint32_t idx = 0; idx < blasCount; idx++)
  {
    oldBlas[idx] = m_blas[idx];
    m_blas[idx]  = {};

    // New compact allocation of buffer and acceleration structure.
    VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    createInfo.size = compactSizes[idx];
    m_blas[idx]     = res.createAccelKHR(createInfo);

    m_compactSize += compactSizes[idx];

    VkCopyAccelerationStructureInfoKHR copy = {VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
    copy.mode                               = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
    copy.src                                = oldBlas[idx].accel;
    copy.dst                                = m_blas[idx].accel;

    vkCmdCopyAccelerationStructureKHR(cmd, &copy);
  }

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void RendererRayTraceTriangles::updateRayTracingBlas(VkCommandBuffer cmd, Resources& res, Scene& scene, float rebuildFraction)
{
  auto blasCount = static_cast<uint32_t>(m_blas.size());

  uint32_t firstIndex = 0;
  uint32_t lastIndex  = blasCount - 1;

  if(rebuildFraction < 1.f)
  {
    uint32_t rebuildCount             = uint32_t(blasCount * rebuildFraction);
    firstIndex                        = m_currentBlasRebuildInstanceStart;
    lastIndex                         = (rebuildCount + firstIndex) % (std::max(1u, blasCount - 1u));
    m_currentBlasRebuildInstanceStart = (lastIndex + 1) % (std::max(1u, blasCount - 1u));
  }

  VkDeviceAddress scratchOffset = 0;

  for(uint32_t idx = 0; idx < blasCount; idx++)
  {

    if((firstIndex <= lastIndex && idx >= firstIndex && idx <= lastIndex)
       || (firstIndex > lastIndex && (idx <= lastIndex || idx >= firstIndex)))
    {
      m_blasBuildInfos[idx].mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      m_blasBuildInfos[idx].srcAccelerationStructure = VK_NULL_HANDLE;
    }
    else
    {
      m_blasBuildInfos[idx].mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      m_blasBuildInfos[idx].srcAccelerationStructure = m_blas[idx].accel;
    }

    if(scratchOffset + m_blasSizeInfos[idx].buildScratchSize >= m_blasScratchBuffer.info.range)
    {
      // Since the scratch buffer is reused across builds, we need a barrier when we start a new batch of builds
      VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

      scratchOffset = 0;
    }

    m_blasBuildInfos[idx].scratchData.deviceAddress = m_blasScratchBuffer.address + scratchOffset;
    scratchOffset +=
        nvh::align_up(m_blasSizeInfos[idx].buildScratchSize, m_accProperties.minAccelerationStructureScratchOffsetAlignment);

    // Building the bottom-level-acceleration-structure
    VkAccelerationStructureBuildRangeInfoKHR* rangeInfo = &m_blasRangeInfos[idx];
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &m_blasBuildInfos[idx], &rangeInfo);
  }

  {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }
}

void RendererRayTraceTriangles::initRayTracingPipeline(Resources& res)
{
  m_dsetContainer.init(res.m_device);

  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

  m_dsetContainer.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  m_dsetContainer.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  m_dsetContainer.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  m_dsetContainer.addBinding(BINDINGS_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, stageFlags);
  m_dsetContainer.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  m_dsetContainer.initLayout();

  VkPushConstantRange pushRange;
  pushRange.offset     = 0;
  pushRange.size       = sizeof(uint32_t);
  pushRange.stageFlags = stageFlags;
  m_dsetContainer.initPipeLayout(1, &pushRange);

  m_dsetContainer.initPool(1);
  std::array<VkWriteDescriptorSet, 5> writeSets;
  writeSets[0] = m_dsetContainer.makeWrite(0, BINDINGS_FRAME_UBO, &res.m_common.view.info);
  writeSets[1] = m_dsetContainer.makeWrite(0, BINDINGS_READBACK_SSBO, &res.m_common.readbackDevice.info);
  writeSets[2] = m_dsetContainer.makeWrite(0, BINDINGS_RENDERINSTANCES_SSBO, &m_renderInstanceBuffer.info);

  VkWriteDescriptorSetAccelerationStructureKHR accelInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  accelInfo.accelerationStructureCount = 1;
  VkAccelerationStructureKHR accel     = m_tlas.accel;
  accelInfo.pAccelerationStructures    = &accel;
  writeSets[3]                         = m_dsetContainer.makeWrite(0, BINDINGS_TLAS, &accelInfo);

  VkDescriptorImageInfo renderTargetInfo;
  renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  renderTargetInfo.imageView   = res.m_framebuffer.viewColor;
  writeSets[4]                 = m_dsetContainer.makeWrite(0, BINDINGS_RENDER_TARGET, &renderTargetInfo);

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);


  nvvkhl::PipelineContainer& p = m_rtPipe;
  p.plines.resize(1);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  for(auto& s : stages)
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

  stages[eRaygen].module     = res.m_shaderManager.get(m_shaders.rayGenShader);
  stages[eRaygen].pName      = "main";
  stages[eRaygen].stage      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eMiss].module       = res.m_shaderManager.get(m_shaders.missShader);
  stages[eMiss].pName        = "main";
  stages[eMiss].stage        = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMissAO].module     = res.m_shaderManager.get(m_shaders.missShaderAO);
  stages[eMissAO].pName      = "main";
  stages[eMissAO].stage      = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eClosestHit].module = res.m_shaderManager.get(m_shaders.closestHitShader);
  stages[eClosestHit].pName  = "main";
  stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss AO
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  //const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> dsetLayouts = {m_dsetContainer.getLayout()};  // , m_pContainer[eGraphic].dstLayout};
  VkPipelineLayoutCreateInfo layoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutCreateInfo.setLayoutCount         = static_cast<uint32_t>(dsetLayouts.size());
  layoutCreateInfo.pSetLayouts            = dsetLayouts.data();
  layoutCreateInfo.pushConstantRangeCount = 0;  //1;
  //pipeline_layout_create_info.pPushConstantRanges    = &push_constant,

  vkCreatePipelineLayout(res.m_device, &layoutCreateInfo, nullptr, &p.layout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  pipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages                      = stages.data();
  pipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
  pipelineInfo.pGroups                      = shaderGroups.data();
  pipelineInfo.maxPipelineRayRecursionDepth = 2;
  pipelineInfo.layout                       = p.layout;

  vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &pipelineInfo, nullptr, &p.plines[0]);

  // Creating the SBT
  m_rtSbt.setup(res.m_device, res.m_queueFamily, &res.m_allocator, m_rtProperties);
  m_rtSbt.create(p.plines[0], pipelineInfo);
}


void RendererRayTraceTriangles::deinit(Resources& res)
{
  deinitBasics(res);

  res.destroy(m_blasScratchBuffer);

  for(auto& b : m_blas)
  {
    res.destroy(b);
  }

  res.destroy(m_tlasInstancesBuffer);
  res.destroy(m_tlasScratchBuffer);
  res.destroy(m_tlas);


  m_rtSbt.destroy();               // Shading binding table wrapper
  m_rtPipe.destroy(res.m_device);  // Hold pipelines and layout

  res.destroyShaders(m_shaders);

  m_dsetContainer.deinit();
}

std::unique_ptr<Renderer> makeRendererRayTraceTriangles()
{
  return std::make_unique<RendererRayTraceTriangles>();
}

void RendererRayTraceTriangles::updatedFrameBuffer(Resources& res)
{
  vkDeviceWaitIdle(res.m_device);
  std::array<VkWriteDescriptorSet, 1> writeSets;
  VkDescriptorImageInfo               renderTargetInfo;
  renderTargetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  renderTargetInfo.imageView   = res.m_framebuffer.viewColor;
  writeSets[0]                 = m_dsetContainer.makeWrite(0, BINDINGS_RENDER_TARGET, &renderTargetInfo);

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
}

}  // namespace animatedclusters
