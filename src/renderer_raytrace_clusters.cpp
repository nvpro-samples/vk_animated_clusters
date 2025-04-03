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
#include <nvh/parallel_work.hpp>
#include <nvh/misc.hpp>

#include "renderer.hpp"
#include "vk_nv_cluster_acc.h"

//////////////////////////////////////////////////////////////////////////

// these settings are for debugging / temporary workarounds

#define DEBUG_SPLIT_INIT 1

//////////////////////////////////////////////////////////////////////////

namespace animatedclusters {

class RendererRayTraceClusters : public Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res);

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  void updateRayTracingScene(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler);
  void updateRayTracingClusters(VkCommandBuffer cmd, Resources& res, Scene& scene);
  void updateRayTracingBlas(VkCommandBuffer cmd, Resources& res, Scene& scene);

  bool initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingTemplates(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingTemplateInstantiations(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingClusters(Resources& res, Scene& scene, const RendererConfig& config);
  bool initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config);

  void initRayTracingPipeline(Resources& res);

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper          m_rtSbt;   // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout

  struct Shaders
  {
    nvvk::ShaderModuleID rayGenShader;
    nvvk::ShaderModuleID closestHitShader;
    nvvk::ShaderModuleID missShader;
    nvvk::ShaderModuleID missShaderAO;
    nvvk::ShaderModuleID computeStatistics;
  } m_shaders;

  VkPipeline       m_computeStatisticsPipeline = nullptr;
  VkPipelineLayout m_computeStatisticsLayout   = nullptr;

  nvvk::DescriptorSetContainer m_dsetContainer;

  // auxiliary rendering data

  uint32_t m_numTotalClusters = 0;

  VkClusterAccelerationStructureTriangleClusterInputNV     m_clusterTriangleInput;
  VkClusterAccelerationStructureClustersBottomLevelInputNV m_clusterBlasInput;

  struct GeometryTemplate
  {
    RBuffer               templatesBuffer;
    std::vector<uint64_t> templateAddresses;
    std::vector<uint32_t> instantionOffsets;
    uint32_t              sumInstantionSizes;
  };

  struct RenderInstanceClusterData
  {
    RBuffer clusterBuffer;
  };

  std::vector<GeometryTemplate>          m_geometryTemplates;
  std::vector<RenderInstanceClusterData> m_renderInstanceClusters;

  // we pre-compute this once for all instances
  // given we use predetermined allocation sizes.
  //
  // args to build clusters for entire scene
  // pre-computed, explicit mode

  // if we use templates
  RBuffer m_instantiationInfoBuffer;  // explicit instantiation src
  // otherwise
  RBuffer m_clusterBuildInfoBuffer;  // implicit cluster build src
  // both store resulting clusters here, this is fed into
  // blas build
  RBuffer m_clusterDstBuffer;   // explicit cluster dst for instantiation, otherwise implicit dst
  RBuffer m_clusterSizeBuffer;  // just for statistics

  // updated every frame
  RBuffer      m_clusterBuffer;                        // cluster dst content
  RLargeBuffer m_clusterBlasBuffer;                    // blas dst content
  RBuffer      m_renderInstanceClusterBlasInfoBuffer;  // blas build src
  RBuffer      m_renderInstanceClusterBlasSizeBuffer;  // just for statistics

  VkDeviceSize m_scratchSize = 0;
  RBuffer      m_scratchBuffer;
};

bool RendererRayTraceClusters::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  std::string prepend = nvh::stringFormat("#define CLUSTER_DEDICATED_VERTICES %d\n", scene.m_config.clusterDedicatedVertices);

  // do shaders first, most likely to contain errors
  m_shaders.rayGenShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl");
  m_shaders.closestHitShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                                                      "render_raytrace_clusters.rchit.glsl", prepend);
  m_shaders.missShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                                "#define RAYTRACING_PAYLOAD_INDEX 0\n");
  m_shaders.missShaderAO = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl",
                                                                  "#define RAYTRACING_PAYLOAD_INDEX 1\n");
  m_shaders.computeStatistics = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "cluster_statistics.comp.glsl");

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRayTraceClusters::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_config = config;

  if(!initShaders(res, scene, config))
    return false;

  initBasics(res, scene, config);

  m_resourceUsageInfo.sceneMemBytes += scene.m_sceneClusterMemBytes;

  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m_rtProperties};
  vkGetPhysicalDeviceProperties2(res.m_physical, &prop2);

  if(!initRayTracingScene(res, scene, config))
  {
    return false;
  }

  initRayTracingPipeline(res);

  {
    VkPushConstantRange pushRange;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(shaderio::StatisticsConstants);
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pPushConstantRanges        = &pushRange;
    layoutInfo.pushConstantRangeCount     = 1;
    vkCreatePipelineLayout(res.m_device, &layoutInfo, nullptr, &m_computeStatisticsLayout);

    VkComputePipelineCreateInfo compInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.stage                       = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                 = "main";
    compInfo.layout                      = m_computeStatisticsLayout;
    compInfo.flags                       = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;

    compInfo.stage.module = res.m_shaderManager.get(m_shaders.computeStatistics);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_computeStatisticsPipeline);
  }

  return true;
}

void RendererRayTraceClusters::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  if(m_config.doAnimation)
  {
    updateAnimation(primary, res, scene, frame, profiler);
    {
      auto timerSection = profiler.timeRecurring("AS Build/Refit", primary);
      updateRayTracingScene(primary, res, scene, frame, profiler);
    }
  }

  vkCmdUpdateBuffer(primary, res.m_common.view.buffer, 0, sizeof(shaderio::FrameConstants), (const uint32_t*)&frame.frameConstants);
  vkCmdFillBuffer(primary, res.m_common.readbackDevice.buffer, 0, sizeof(shaderio::Readback), 0);

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

  {
    // statistics
    shaderio::StatisticsConstants statisticsConstants;

    // over all clusters
    vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_computeStatisticsPipeline);


    uint32_t numClusters      = uint32_t(m_clusterSizeBuffer.info.range / sizeof(uint32_t));
    statisticsConstants.count = numClusters;
    statisticsConstants.sizes = m_clusterSizeBuffer.address;
    statisticsConstants.sum   = res.m_common.readbackDevice.address + offsetof(shaderio::Readback, clustersSize);

    vkCmdPushConstants(primary, m_computeStatisticsLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(shaderio::StatisticsConstants), &statisticsConstants);
    vkCmdDispatch(primary, (numClusters + STATISTICS_WORKGROUP_SIZE - 1) / STATISTICS_WORKGROUP_SIZE, 1, 1);

    uint32_t numInstances     = uint32_t(m_renderInstanceClusterBlasSizeBuffer.info.range / sizeof(uint32_t));
    statisticsConstants.count = numInstances;
    statisticsConstants.sizes = m_renderInstanceClusterBlasSizeBuffer.address;
    statisticsConstants.sum   = res.m_common.readbackDevice.address + offsetof(shaderio::Readback, blasesSize);

    vkCmdPushConstants(primary, m_computeStatisticsLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(shaderio::StatisticsConstants), &statisticsConstants);
    vkCmdDispatch(primary, (numInstances + STATISTICS_WORKGROUP_SIZE - 1) / STATISTICS_WORKGROUP_SIZE, 1, 1);
  }
}

void RendererRayTraceClusters::updateRayTracingScene(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  // wait for animation update
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run template instantiation or clas build
  {
    auto timerSection = profiler.timeRecurring("clas", cmd);
    updateRayTracingClusters(cmd, res, scene);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run blas build
  {
    auto timerSection = profiler.timeRecurring("blas", cmd);
    updateRayTracingBlas(cmd, res, scene);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run tlas build/update
  {
    auto timerSection = profiler.timeRecurring("tlas", cmd);
    updateRayTracingTlas(cmd, res, scene, !frame.forceTlasFullRebuild);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);
}

void RendererRayTraceClusters::updateRayTracingClusters(VkCommandBuffer cmd, Resources& res, Scene& scene)
{
  VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

  if(m_config.useTemplates)
  {
    // setup instantiation inputs
    inputs.maxAccelerationStructureCount = m_numTotalClusters;
    inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.opInput.pTriangleClusters     = &m_clusterTriangleInput;
    inputs.flags                         = m_config.templateInstantiateFlags;

    cmdInfo.dstAddressesArray.deviceAddress = m_clusterDstBuffer.address;
    cmdInfo.dstAddressesArray.size          = m_clusterDstBuffer.info.range;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmdInfo.dstSizesArray.deviceAddress = m_clusterSizeBuffer.address;
    cmdInfo.dstSizesArray.size          = m_clusterSizeBuffer.info.range;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = m_instantiationInfoBuffer.address;
    cmdInfo.srcInfosArray.size          = m_instantiationInfoBuffer.info.range;
    cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);

    cmdInfo.scratchData = m_scratchBuffer.address;
    cmdInfo.input       = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
  }
  else
  {
    // setup cluster build inputs
    inputs.maxAccelerationStructureCount = m_numTotalClusters;

    // use implicit if we don't have per render instance cluster buffers
    inputs.opMode = m_renderInstanceClusters.empty() ? VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV :
                                                       VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    inputs.flags                     = m_config.clusterBuildFlags;

    cmdInfo.dstImplicitData = m_clusterBuffer.address;  // can be zero if explicit

    cmdInfo.dstAddressesArray.deviceAddress = m_clusterDstBuffer.address;
    cmdInfo.dstAddressesArray.size          = m_clusterDstBuffer.info.range;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmdInfo.dstSizesArray.deviceAddress = m_clusterSizeBuffer.address;
    cmdInfo.dstSizesArray.size          = m_clusterSizeBuffer.info.range;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = m_clusterBuildInfoBuffer.address;
    cmdInfo.srcInfosArray.size          = m_clusterBuildInfoBuffer.info.range;
    cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV);

    cmdInfo.scratchData = m_scratchBuffer.address;
    cmdInfo.input       = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
  }
}

void RendererRayTraceClusters::updateRayTracingBlas(VkCommandBuffer cmd, Resources& res, Scene& scene)
{
  VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};

  // setup blas inputs
  inputs.maxAccelerationStructureCount = uint32_t(m_renderInstances.size());
  inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel  = &m_clusterBlasInput;
  inputs.flags                         = m_config.clusterBlasFlags;

  // we feed the generated blas addresses directly into the ray instances
  cmdInfo.dstAddressesArray.deviceAddress =
      m_tlasInstancesBuffer.address + offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference);
  cmdInfo.dstAddressesArray.size   = m_tlasInstancesBuffer.info.range;
  cmdInfo.dstAddressesArray.stride = sizeof(VkAccelerationStructureInstanceKHR);

  cmdInfo.dstSizesArray.deviceAddress = m_renderInstanceClusterBlasSizeBuffer.address;
  cmdInfo.dstSizesArray.size          = m_renderInstanceClusterBlasSizeBuffer.info.range;
  cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

  cmdInfo.srcInfosArray.deviceAddress = m_renderInstanceClusterBlasInfoBuffer.address;
  cmdInfo.srcInfosArray.size          = m_renderInstanceClusterBlasInfoBuffer.info.range;
  cmdInfo.srcInfosArray.stride        = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);

  // in implicit mode we provide one big chunk from which outputs are sub-allocated
  cmdInfo.dstImplicitData = m_clusterBlasBuffer.address;

  cmdInfo.scratchData = m_scratchBuffer.address;
  cmdInfo.input       = inputs;
  vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);
}

bool RendererRayTraceClusters::initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config)
{
  // used for cluster builds or instantiations
  // which do entire scene at once
  m_clusterTriangleInput              = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  m_clusterTriangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  m_clusterTriangleInput.maxClusterTriangleCount     = scene.m_config.clusterTriangles;
  m_clusterTriangleInput.maxClusterVertexCount       = scene.m_config.clusterVertices;
  m_clusterTriangleInput.maxTotalTriangleCount       = 0;
  m_clusterTriangleInput.maxTotalVertexCount         = 0;
  m_clusterTriangleInput.minPositionTruncateBitCount = config.positionTruncateBits;

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    const Scene::Geometry&          geometry       = scene.m_geometries[renderInstance.geometryID];
    m_clusterTriangleInput.maxTotalTriangleCount += geometry.numTriangles;
    m_clusterTriangleInput.maxTotalVertexCount += uint32_t(geometry.clusterLocalVertices.size());
    m_numTotalClusters += geometry.numClusters;
  }

  if(config.useTemplates)
  {
    initRayTracingTemplates(res, scene, config);
    initRayTracingTemplateInstantiations(res, scene, config);
  }
  else
  {
    initRayTracingClusters(res, scene, config);
  }

  // BLAS creation
  if(!initRayTracingBlas(res, scene, config))
  {
    deinit(res);
    return false;
  }

  // TLAS creation
  initRayTracingTlas(res, scene, config);

  m_scratchBuffer = res.createBuffer(m_scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_resourceUsageInfo.rtOtherMemBytes += m_scratchBuffer.info.range;

  // initial build
  {
    VkCommandBuffer cmd = res.createTempCmdBuffer();

    updateRayTracingClusters(cmd, res, scene);

    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    updateRayTracingBlas(cmd, res, scene);


    memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                               | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

    updateRayTracingTlas(cmd, res, scene);

    printf("submitting cluster init\n");
    res.tempSyncSubmit(cmd);
    printf("done\n");
  }

  return true;
}

void RendererRayTraceClusters::initRayTracingTemplates(Resources& res, Scene& scene, const RendererConfig& config)
{
  // This function generates templates for every geometry
  // and figures out the instantiation size for each cluster.
  //
  // Storage space for the instantiated CLAS is setup within `RendererRayTraceClusters::initRayTracingTemplateInstantiations`

  assert(config.useTemplates);

  m_geometryTemplates.resize(scene.m_geometries.size());

  bool useImplicitTemplates = config.useImplicitTemplates;

  RBuffer implicitBuffer;

  // we use the same scratch buffer for various operations
  VkDeviceSize tempScratchSize = 0;

  // slightly lower totals because we do one geometry at a time for template builds.
  VkClusterAccelerationStructureTriangleClusterInputNV templateTriangleInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  templateTriangleInput.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
  templateTriangleInput.maxClusterTriangleCount     = scene.m_config.clusterTriangles;
  templateTriangleInput.maxClusterVertexCount       = scene.m_config.clusterVertices;
  templateTriangleInput.maxTotalTriangleCount       = scene.m_maxPerGeometryTriangles;
  templateTriangleInput.maxTotalVertexCount         = scene.m_maxPerGeometryClusterVertices;
  templateTriangleInput.minPositionTruncateBitCount = config.positionTruncateBits;

  VkClusterAccelerationStructureMoveObjectsInputNV moveInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_MOVE_OBJECTS_INPUT_NV};

  // following operations are done per cluster in advance
  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = scene.m_maxPerGeometryClusters;
  inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
  inputs.opMode = useImplicitTemplates ? VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV :
                                         VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
  inputs.opInput.pTriangleClusters                   = &templateTriangleInput;
  inputs.flags                                       = config.templateBuildFlags;
  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);

  if(useImplicitTemplates)
  {
    implicitBuffer = res.createBuffer(sizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                                                               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                               | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // implicit builds are not guaranteed to be perfectly compact either, we run an extra compaction step after the implicit build.
    moveInput.type              = VK_CLUSTER_ACCELERATION_STRUCTURE_TYPE_TRIANGLE_CLUSTER_TEMPLATE_NV;
    moveInput.noMoveOverlap     = VK_TRUE;  // we move/copy from implicitBuffer to final per-geometry buffer
    moveInput.maxMovedBytes     = sizesInfo.accelerationStructureSize;  // worst case everything is moved
    inputs.opType               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
    inputs.opMode               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.flags                = 0;
    inputs.opInput.pMoveObjects = &moveInput;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    tempScratchSize = std::max(tempScratchSize, sizesInfo.updateScratchSize);
  }
  else
  {
    // when not doing implicit build, we want to query the sizes in advance.
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.flags  = config.templateBuildFlags;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);
  }

  // to know how big the clusters will be after instantiation we query their s
  inputs.opInput.pTriangleClusters = &templateTriangleInput;
  inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
  inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
  inputs.flags                     = config.templateInstantiateFlags;
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  tempScratchSize = std::max(tempScratchSize, sizesInfo.buildScratchSize);

  // let's setup temporary resources

  RBuffer scratchBuffer = res.createBuffer(tempScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  std::vector<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV> templateInfos(scene.m_maxPerGeometryClusters);
  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiateInfos(scene.m_maxPerGeometryClusters);

  size_t infoSize = std::max(std::max(sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV),
                                      sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV)),
                             sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV));

  RBuffer infosBuffer = res.createBuffer(infoSize * templateInfos.size(),
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer sizesBuffer = res.createBuffer(sizeof(uint32_t) * instantiateInfos.size(),
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  RBuffer dstAddressesBuffer = res.createBuffer(sizeof(uint64_t) * instantiateInfos.size(),
                                                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);


  // 32 byte alignment requirement for bbox
  struct TemplateBbox
  {
    shaderio::BBox bbox;
    uint32_t       _pad[2];
  };

  RBuffer bboxesBuffer = res.createBuffer(sizeof(TemplateBbox) * instantiateInfos.size(),
                                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  for(size_t g = 0; g < scene.m_geometries.size(); g++)
  {
    GeometryTemplate&      geometryTemplate = m_geometryTemplates[g];
    const Scene::Geometry& geometry         = scene.m_geometries[g];

    uint32_t numClusters = uint32_t(geometry.clusters.size());

    float bloatSize = glm::length(geometry.bbox.hi - geometry.bbox.lo) * config.templateBBoxBloat;

    auto* templateInfosMapping =
        reinterpret_cast<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV*>(infosBuffer.mapping);

    for(uint32_t c = 0; c < numClusters; c++)
    {
      const shaderio::Cluster&                                          cluster      = geometry.clusters[c];
      VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV& templateInfo = templateInfosMapping[c];

      // add bloat to original bbox

      TemplateBbox& tempBbox = ((TemplateBbox*)bboxesBuffer.mapping)[c];

      shaderio::BBox clusterBbox = geometry.clusterBboxes[c];
      clusterBbox.lo -= bloatSize;
      clusterBbox.hi += bloatSize;

      tempBbox.bbox = clusterBbox;

      templateInfo = {0};

      templateInfo.clusterID     = c;
      templateInfo.vertexCount   = cluster.numVertices;
      templateInfo.triangleCount = cluster.numTriangles;
      templateInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;

      if(scene.m_config.clusterDedicatedVertices)
      {
        templateInfo.indexBuffer = geometry.clusterLocalTrianglesBuffer.address + (sizeof(uint8_t) * cluster.firstLocalTriangle);
        templateInfo.indexBufferStride = sizeof(uint8_t);
        templateInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

        templateInfo.vertexBuffer = geometry.positionsBuffer.address + (sizeof(glm::vec3) * cluster.firstLocalVertex);
        templateInfo.vertexBufferStride = sizeof(glm::vec3);
      }
      else
      {
        templateInfo.indexBuffer = geometry.trianglesBuffer.address + (sizeof(uint32_t) * cluster.firstTriangle * 3);
        templateInfo.indexBufferStride = sizeof(uint32_t);
        templateInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV;

        templateInfo.vertexBuffer       = geometry.positionsBuffer.address;
        templateInfo.vertexBufferStride = sizeof(glm::vec3);
      }

      templateInfo.positionTruncateBitCount = config.positionTruncateBits;

      templateInfo.instantiationBoundingBoxLimit =
          config.templateBBoxBloat < 0 ? 0 : bboxesBuffer.address + sizeof(TemplateBbox) * c;
    }

    // actual count of current geometry
    inputs.maxAccelerationStructureCount = numClusters;

    VkCommandBuffer cmd;
    VkClusterAccelerationStructureCommandsInfoNV cmdInfo = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    cmdInfo.srcInfosArray.deviceAddress     = infosBuffer.address;
    cmdInfo.srcInfosArray.size              = infosBuffer.info.range;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
    cmdInfo.dstSizesArray.size              = sizesBuffer.info.range;
    cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
    cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
    cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.info.range;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);
    cmdInfo.scratchData                     = scratchBuffer.address;

    if(useImplicitTemplates)
    {
      inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
      inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
      inputs.flags  = config.templateBuildFlags;

      cmdInfo.dstImplicitData = implicitBuffer.address;
    }
    else
    {
      // query size of templates
      inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
      inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV;
      inputs.flags  = config.templateBuildFlags;
    }

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // compute template buffer sizes

    uint32_t buildSum = 0;
    for(uint32_t c = 0; c < numClusters; c++)
    {
      buildSum += ((const uint32_t*)sizesBuffer.mapping)[c];
    }
    // allocate outputs and setup dst addresses
    geometryTemplate.templatesBuffer = res.createBuffer(buildSum, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
    m_resourceUsageInfo.rtOtherMemBytes += buildSum;

    geometryTemplate.templateAddresses.resize(numClusters);

    if(useImplicitTemplates)
    {
      // after the implicit build, let's move from the scratch implicit buffer
      // to the final per-geometry buffer in a compacted fashion.

      // compute address / offset for each template
      uint64_t* dstAddresses = ((uint64_t*)dstAddressesBuffer.mapping);
      buildSum               = 0;

      auto* moveInfosMapping = reinterpret_cast<VkClusterAccelerationStructureMoveObjectsInfoNV*>(infosBuffer.mapping);

      for(uint32_t c = 0; c < numClusters; c++)
      {
        geometryTemplate.templateAddresses[c] = geometryTemplate.templatesBuffer.address + buildSum;
        uint32_t templateSize                 = ((const uint32_t*)sizesBuffer.mapping)[c];

        // read from old address
        moveInfosMapping[c].srcAccelerationStructure = dstAddresses[c];
        // setup new dst address
        dstAddresses[c] = geometryTemplate.templateAddresses[c];

        assert(templateSize);

        buildSum += templateSize;
      }

      cmd = res.createTempCmdBuffer();

      inputs.opType               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_MOVE_OBJECTS_NV;
      inputs.opMode               = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
      inputs.flags                = 0;
      inputs.opInput.pMoveObjects = &moveInput;

      cmdInfo.srcInfosArray.deviceAddress     = infosBuffer.address;
      cmdInfo.srcInfosArray.size              = infosBuffer.info.range;
      cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV);
      cmdInfo.dstSizesArray.deviceAddress     = 0;
      cmdInfo.dstSizesArray.size              = 0;
      cmdInfo.dstSizesArray.stride            = 0;
      cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
      cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.info.range;
      cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

      cmdInfo.input = inputs;
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

      res.tempSyncSubmit(cmd);
    }
    else
    {
      uint64_t* dstAddresses = ((uint64_t*)dstAddressesBuffer.mapping);
      buildSum               = 0;
      for(uint32_t c = 0; c < numClusters; c++)
      {
        dstAddresses[c]                       = geometryTemplate.templatesBuffer.address + buildSum;
        geometryTemplate.templateAddresses[c] = geometryTemplate.templatesBuffer.address + buildSum;
        buildSum += ((const uint32_t*)sizesBuffer.mapping)[c];
      }

      // build explicit
      inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;

      cmd = res.createTempCmdBuffer();

      cmdInfo.input = inputs;
      vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

      res.tempSyncSubmit(cmd);
    }

    // now compute instantiation sizes
    geometryTemplate.instantionOffsets.resize(numClusters);

    auto* instantiationInfosMapping =
        reinterpret_cast<VkClusterAccelerationStructureInstantiateClusterInfoNV*>(infosBuffer.mapping);

    for(uint32_t c = 0; c < numClusters; c++)
    {
      const shaderio::Cluster&                                cluster           = geometry.clusters[c];
      VkClusterAccelerationStructureInstantiateClusterInfoNV& instantiationInfo = instantiationInfosMapping[c];

      instantiationInfo.clusterIdOffset        = 0;
      instantiationInfo.clusterTemplateAddress = geometryTemplate.templateAddresses[c];
      instantiationInfo.geometryIndexOffset    = 0;
      // leave vertices off given we are looking for worst case instantiation size, not actual
      instantiationInfo.vertexBuffer.startAddress  = 0;
      instantiationInfo.vertexBuffer.strideInBytes = 0;
    }

    // query size of instantiations
    inputs.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_COMPUTE_SIZES_NV;
    inputs.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.flags  = config.templateInstantiateFlags;

    cmdInfo.srcInfosArray.deviceAddress     = infosBuffer.address;
    cmdInfo.srcInfosArray.size              = infosBuffer.info.range;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
    cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
    cmdInfo.dstSizesArray.size              = sizesBuffer.info.range;
    cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
    cmdInfo.dstAddressesArray.deviceAddress = 0;
    cmdInfo.dstAddressesArray.size          = 0;
    cmdInfo.dstAddressesArray.stride        = 0;

    cmd = res.createTempCmdBuffer();

    cmdInfo.input = inputs;
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &cmdInfo);

    res.tempSyncSubmit(cmd);

    // compute output offsets for instantiations, and total sum
    // this is later used for building the per-instance clusters
    geometryTemplate.instantionOffsets.resize(numClusters);

    uint32_t instantiationSum = 0;
    for(uint32_t c = 0; c < numClusters; c++)
    {
      geometryTemplate.instantionOffsets[c] = instantiationSum;
      uint32_t instantiationSize            = ((const uint32_t*)sizesBuffer.mapping)[c];
      assert(instantiationSize);
      instantiationSum += instantiationSize;
    }

    geometryTemplate.sumInstantionSizes = instantiationSum;
  }

  // delete temp resources
  res.destroy(scratchBuffer);
  res.destroy(infosBuffer);
  res.destroy(sizesBuffer);
  res.destroy(dstAddressesBuffer);
  res.destroy(bboxesBuffer);
  res.destroy(implicitBuffer);
}

void RendererRayTraceClusters::initRayTracingTemplateInstantiations(Resources& res, Scene& scene, const RendererConfig& config)
{
  // After we built the templates we now allocate destination space for the CLAS that are generated during
  // template instantiation.
  // We also upload the information for the instantiation step. At runtime we only have to update the vertex
  // buffers and can the run the build from these pre-generated inputs.

  assert(config.useTemplates);

  {
    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = 1;
    inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_INSTANTIATE_TRIANGLE_CLUSTER_NV;
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    inputs.flags                     = 0;
    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);
  }

  // for every instance we create its own cluster buffer (allows us to more easily circumvent 4 GB buffer size limitations)

  m_renderInstanceClusters.resize(m_renderInstances.size());

  size_t instantionSize = 0;
  size_t numClusters    = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    uint32_t geometryID = m_renderInstances[i].geometryID;

    m_renderInstanceClusters[i].clusterBuffer =
        res.createBuffer(m_geometryTemplates[geometryID].sumInstantionSizes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceUsageInfo.rtClasMemBytes += m_renderInstanceClusters[i].clusterBuffer.info.range;

    numClusters += m_geometryTemplates[geometryID].instantionOffsets.size();
  }

  // the actual instantiation process gets argument buffers for the entire scene

  m_instantiationInfoBuffer =
      res.createBuffer(sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * numClusters,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_instantiationInfoBuffer.info.range;

  m_clusterDstBuffer = res.createBuffer(sizeof(uint64_t) * numClusters,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterDstBuffer.info.range;

  m_clusterSizeBuffer = res.createBuffer(sizeof(uint32_t) * numClusters,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterSizeBuffer.info.range;

  // fill instantiation task data
  // for now on cpu, more typically this would be done on GPU based on culling etc.
  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiationInfos(numClusters, {0});
  std::vector<uint64_t>                                               instantiationDst(numClusters, {0});

  size_t clusterOffset = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    const shaderio::RenderInstance&  renderInstance        = m_renderInstances[i];
    const RenderInstanceClusterData& renderInstanceCluster = m_renderInstanceClusters[i];
    GeometryTemplate&                geometryTemplate      = m_geometryTemplates[renderInstance.geometryID];
    const Scene::Geometry&           geometry              = scene.m_geometries[renderInstance.geometryID];


    // setup instantiation

    uint64_t baseAddress = renderInstanceCluster.clusterBuffer.address;

    for(size_t c = 0; c < geometry.clusters.size(); c++)
    {
      const shaderio::Cluster& cluster = geometry.clusters[c];

      // input
      VkClusterAccelerationStructureInstantiateClusterInfoNV& instInfo = instantiationInfos[clusterOffset + c];

      instInfo.clusterIdOffset        = 0;  // stored in template
      instInfo.clusterTemplateAddress = geometryTemplate.templateAddresses[c];

      if(scene.m_config.clusterDedicatedVertices)
      {
        instInfo.vertexBuffer.startAddress  = renderInstance.positions + (sizeof(glm::vec3) * cluster.firstLocalVertex);
        instInfo.vertexBuffer.strideInBytes = sizeof(glm::vec3);
      }
      else
      {
        // we don't use per-cluster vertices

        instInfo.vertexBuffer.startAddress  = renderInstance.positions;
        instInfo.vertexBuffer.strideInBytes = sizeof(glm::vec3);
      }


      // destination
      uint32_t instOffset                 = geometryTemplate.instantionOffsets[c];
      uint64_t clusterAddress             = baseAddress + instOffset;
      instantiationDst[clusterOffset + c] = clusterAddress;
    }


    clusterOffset += geometry.clusters.size();
  }

  // these buffers are used during the CLAS build process in `RendererRayTraceClusters::updateRayTracingClusters`

  res.simpleUploadBuffer(m_instantiationInfoBuffer, instantiationInfos.data());
  res.simpleUploadBuffer(m_clusterDstBuffer, instantiationDst.data());
}

void RendererRayTraceClusters::initRayTracingClusters(Resources& res, Scene& scene, const RendererConfig& config)
{
  // When not using templates we upload the CLAS build arguments for all runtime generated CLAS.
  // During CLAS build they re-use the same index buffers, but are fetching the vertices for
  // each instance individually.
  // The output space for the CLAS is based on the worst-case size of the largest cluster and
  // we pre-configure the CLAS destination addresses accordingly.
  // This will take more memory than the template way, which allowed us to query the size of the
  // individual template in advance, independent of the final vertex positions.

  assert(!config.useTemplates);

  VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
  inputs.maxAccelerationStructureCount             = m_numTotalClusters;
  inputs.opType                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
  inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
  inputs.flags                     = config.clusterBuildFlags;
  VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
  m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);

  // We can build clusters for the entire scene either in implicit or explicit mode.
  // Implicit requires that we have one single destination buffer for all clusters.
  // This however can cause issues with max buffer size.

  bool useExplicit =
      sizesInfo.accelerationStructureSize > std::min(res.m_context->m_physicalInfo.properties11.maxMemoryAllocationSize,
                                                     res.m_context->m_physicalInfo.properties13.maxBufferSize);

  VkDeviceSize singleExplicitClusterSize = 0;

  if(useExplicit)
  {
    // in explicit we manually distribute the clusters across multiple buffers
    inputs.opMode                    = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    inputs.opInput.pTriangleClusters = &m_clusterTriangleInput;
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);

    // in explicit the returned size is that of one element
    singleExplicitClusterSize = sizesInfo.accelerationStructureSize;

    m_renderInstanceClusters.resize(m_renderInstances.size());

    for(size_t i = 0; i < m_renderInstances.size(); i++)
    {
      uint32_t geometryID = m_renderInstances[i].geometryID;

      m_renderInstanceClusters[i].clusterBuffer =
          res.createBuffer(singleExplicitClusterSize * m_renderInstances[i].numClusters,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      m_resourceUsageInfo.rtClasMemBytes += m_renderInstanceClusters[i].clusterBuffer.info.range;
    }
  }
  else
  {
    m_clusterBuffer = res.createBuffer(sizesInfo.accelerationStructureSize,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                           | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceUsageInfo.rtClasMemBytes += m_clusterBuffer.info.range;
  }

  // in both cases (explicit and implicit) the argument buffers are for the entire scene

  m_clusterBuildInfoBuffer =
      res.createBuffer(sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV) * m_numTotalClusters,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterBuildInfoBuffer.info.range;

  m_clusterDstBuffer = res.createBuffer(sizeof(uint64_t) * m_numTotalClusters,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterDstBuffer.info.range;

  m_clusterSizeBuffer = res.createBuffer(sizeof(uint32_t) * m_numTotalClusters,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterSizeBuffer.info.range;

  // fill build info task data
  // for now on cpu, realistically this is done on GPU based on culling etc.
  std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> buildInfos(m_numTotalClusters, {0});
  std::vector<uint64_t>                                                 buildDsts(useExplicit ? m_numTotalClusters : 0);

  size_t clusterOffset = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    const shaderio::RenderInstance& renderInstance = m_renderInstances[i];

    const Scene::Geometry& geometry = scene.m_geometries[renderInstance.geometryID];

    uint64_t baseAddress = 0;
    if(useExplicit)
    {
      const RenderInstanceClusterData& renderInstanceCluster = m_renderInstanceClusters[i];
      baseAddress                                            = renderInstanceCluster.clusterBuffer.address;
    }

    // setup build

    for(size_t c = 0; c < geometry.clusters.size(); c++)
    {
      const shaderio::Cluster& cluster = geometry.clusters[c];

      // input
      VkClusterAccelerationStructureBuildTriangleClusterInfoNV& buildInfo = buildInfos[clusterOffset + c];

      buildInfo = {0};

      buildInfo.clusterID     = uint32_t(c);
      buildInfo.vertexCount   = cluster.numVertices;
      buildInfo.triangleCount = cluster.numTriangles;

      buildInfo.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;

      if(scene.m_config.clusterDedicatedVertices)
      {
        buildInfo.indexBuffer = geometry.clusterLocalTrianglesBuffer.address + (sizeof(uint8_t) * cluster.firstLocalTriangle);
        buildInfo.indexBufferStride = sizeof(uint8_t);
        buildInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;

        buildInfo.vertexBuffer       = renderInstance.positions + (sizeof(glm::vec3) * cluster.firstLocalVertex);
        buildInfo.vertexBufferStride = sizeof(glm::vec3);
      }
      else
      {
        buildInfo.indexBuffer       = geometry.trianglesBuffer.address + (sizeof(uint32_t) * cluster.firstTriangle * 3);
        buildInfo.indexBufferStride = sizeof(uint32_t);
        buildInfo.indexType         = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_32BIT_NV;

        buildInfo.vertexBuffer       = renderInstance.positions;
        buildInfo.vertexBufferStride = sizeof(glm::vec3);
      }

      buildInfo.positionTruncateBitCount = config.positionTruncateBits;

      if(useExplicit)
      {
        // Explicit requires us to provide the destination of a cluster manually.
        // Simply base this on the worst-case size for each cluster.

        buildDsts[clusterOffset + c] = baseAddress + singleExplicitClusterSize * c;
      }
    }

    clusterOffset += geometry.clusters.size();
  }

  // these buffers are used during the CLAS build process in `RendererRayTraceClusters::updateRayTracingClusters`
  res.simpleUploadBuffer(m_clusterBuildInfoBuffer, buildInfos.data());
  if(useExplicit)
  {
    res.simpleUploadBuffer(m_clusterDstBuffer, buildDsts.data());
  }
}

bool RendererRayTraceClusters::initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config)
{
  // Setting up the BLAS is agnostic to whether template instantiations or regular CLAS builds are used.
  // In both cases we provide a list of the freshly built CLAS references.

  std::vector<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> blasInfos(m_renderInstances.size(), {0});

  size_t clusterOffset = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    const Scene::Geometry&          geometry       = scene.m_geometries[renderInstance.geometryID];

    // setup blas/ray instance
    VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV& blasInfo = blasInfos[i];
    // starting address of array of the dst cluster addresses from instantiation
    // becomes input for blas build
    blasInfo.clusterReferences       = m_clusterDstBuffer.address + sizeof(uint64_t) * clusterOffset;
    blasInfo.clusterReferencesCount  = geometry.numClusters;
    blasInfo.clusterReferencesStride = sizeof(uint64_t);

    clusterOffset += geometry.numClusters;
  }

  // required inputs for blas building and tlas building
  // we will always use implicit mode for this
  m_renderInstanceClusterBlasInfoBuffer =
      res.createBuffer(sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * blasInfos.size(),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_renderInstanceClusterBlasInfoBuffer.info.range;

  m_renderInstanceClusterBlasSizeBuffer =
      res.createBuffer(sizeof(uint32_t) * blasInfos.size(),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_renderInstanceClusterBlasSizeBuffer.info.range;

  res.simpleUploadBuffer(m_renderInstanceClusterBlasInfoBuffer, blasInfos.data());


  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.
  {
    uint32_t numInstances = (uint32_t)blasInfos.size();

    m_clusterBlasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    m_clusterBlasInput.maxClusterCountPerAccelerationStructure = scene.m_maxPerGeometryClusters;
    m_clusterBlasInput.maxTotalClusterCount                    = m_numTotalClusters;

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = numInstances;
    inputs.opMode                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel = &m_clusterBlasInput;
    inputs.flags                        = config.clusterBlasFlags;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);

    m_clusterBlasBuffer =
        res.createLargeBuffer(sizesInfo.accelerationStructureSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_resourceUsageInfo.rtBlasMemBytes += m_clusterBlasBuffer.info.range;
  }

  return true;
}

void RendererRayTraceClusters::initRayTracingPipeline(Resources& res)
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

  // NEW for clusters! we need to enable their usage explicitly for a ray tracing pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
  pipeClusters.allowClusterAccelerationStructure = true;

  // chain extension it into next of the pipeline create info
  pipelineInfo.pNext = &pipeClusters;

  VkResult result = vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &pipelineInfo, nullptr, &p.plines[0]);

  // Creating the SBT
  m_rtSbt.setup(res.m_device, res.m_queueFamily, &res.m_allocator, m_rtProperties);
  m_rtSbt.create(p.plines[0], pipelineInfo);
}

void RendererRayTraceClusters::deinit(Resources& res)
{
  deinitBasics(res);

  for(auto& it : m_geometryTemplates)
  {
    res.destroy(it.templatesBuffer);
  }

  for(auto& it : m_renderInstanceClusters)
  {
    res.destroy(it.clusterBuffer);
  }

  res.destroy(m_renderInstanceClusterBlasInfoBuffer);
  res.destroy(m_renderInstanceClusterBlasSizeBuffer);
  res.destroy(m_clusterBuffer);
  res.destroy(m_clusterBlasBuffer);
  res.destroy(m_clusterDstBuffer);
  res.destroy(m_clusterSizeBuffer);
  res.destroy(m_clusterBuildInfoBuffer);
  res.destroy(m_instantiationInfoBuffer);
  res.destroy(m_scratchBuffer);
  res.destroy(m_tlasInstancesBuffer);
  res.destroy(m_tlasScratchBuffer);
  res.destroy(m_tlas);

  vkDestroyPipeline(res.m_device, m_computeStatisticsPipeline, nullptr);
  vkDestroyPipelineLayout(res.m_device, m_computeStatisticsLayout, nullptr);

  m_rtSbt.destroy();               // Shading binding table wrapper
  m_rtPipe.destroy(res.m_device);  // Hold pipelines and layout

  res.destroyShaders(m_shaders);

  m_dsetContainer.deinit();
}

std::unique_ptr<Renderer> makeRendererRayTraceClusters()
{
  return std::make_unique<RendererRayTraceClusters>();
}

void RendererRayTraceClusters::updatedFrameBuffer(Resources& res)
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
