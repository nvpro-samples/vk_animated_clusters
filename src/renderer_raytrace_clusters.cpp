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

#include <volk.h>

#include <nvvk/acceleration_structures.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/commands.hpp>
#include <fmt/format.h>

#include "renderer.hpp"

//////////////////////////////////////////////////////////////////////////

// these settings are for debugging / temporary workarounds

#define DEBUG_SPLIT_INIT 1

//////////////////////////////////////////////////////////////////////////

namespace animatedclusters {

class RendererRayTraceClusters : public Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;
  virtual void updatedFrameBuffer(Resources& res) override;

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  void updateRayTracingScene(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void updateRayTracingClusters(VkCommandBuffer cmd, Resources& res, Scene& scene);
  void updateRayTracingBlas(VkCommandBuffer cmd, Resources& res, Scene& scene);

  bool initRayTracingScene(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingTemplates(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingTemplateInstantiations(Resources& res, Scene& scene, const RendererConfig& config);
  void initRayTracingClusters(Resources& res, Scene& scene, const RendererConfig& config);
  bool initRayTracingBlas(Resources& res, Scene& scene, const RendererConfig& config);

  void initRayTracingPipeline(Resources& res);

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

  nvvk::SBTGenerator::Regions m_sbtRegions;
  nvvk::Buffer                m_sbtBuffer;

  struct Shaders
  {
    shaderc::SpvCompilationResult rayGen;
    shaderc::SpvCompilationResult rayClosestHit;
    shaderc::SpvCompilationResult rayMiss;
    shaderc::SpvCompilationResult rayMissAO;
    shaderc::SpvCompilationResult computeBlasInstances;
  } m_shaders;

  struct Pipelines
  {
    VkPipeline rayTracing{};
    VkPipeline computeBlasInstances{};
  } m_pipelines;

  nvvk::DescriptorPack m_dsetPack;
  VkPipelineLayout     m_pipelineLayout;
  VkPipelineLayout     m_computePipelineLayout;

  // auxiliary rendering data

  uint32_t m_numTotalClusters = 0;
  uint32_t m_blasCount        = 0;

  VkClusterAccelerationStructureTriangleClusterInputNV     m_clusterTriangleInput;
  VkClusterAccelerationStructureClustersBottomLevelInputNV m_clusterBlasInput;

  struct GeometryTemplate
  {
    nvvk::Buffer          templatesBuffer;
    std::vector<uint64_t> templateAddresses;
    std::vector<uint32_t> instantionOffsets;
    uint32_t              sumInstantionSizes;
  };

  struct RenderInstanceClusterData
  {
    nvvk::Buffer clusterBuffer;
  };

  std::vector<GeometryTemplate>          m_geometryTemplates;
  std::vector<RenderInstanceClusterData> m_renderInstanceClusters;

  // we pre-compute this once for all instances
  // given we use predetermined allocation sizes.
  //
  // args to build clusters for entire scene
  // pre-computed, explicit mode

  // if we use templates
  nvvk::Buffer m_instantiationInfoBuffer;  // explicit instantiation src
  // otherwise
  nvvk::Buffer m_clusterBuildInfoBuffer;  // implicit cluster build src
  // both store resulting clusters here, this is fed into
  // blas build
  nvvk::Buffer m_clusterDstBuffer;   // explicit cluster dst for instantiation, otherwise implicit dst
  nvvk::Buffer m_clusterSizeBuffer;  // just for statistics

  // updated every frame in animation
  nvvk::Buffer      m_clusterBuffer;          // cluster dst content
  nvvk::LargeBuffer m_clusterBlasBuffer;      // blas dst content
  nvvk::Buffer      m_clusterBlasInfoBuffer;  // blas build src
  nvvk::Buffer      m_clusterBlasSizeBuffer;  // just for statistics
  nvvk::Buffer      m_clusterBlasAddressBuffer;

  VkDeviceSize m_scratchSize = 0;
  nvvk::Buffer m_scratchBuffer;

  bool m_forceUpdate = true;
};

bool RendererRayTraceClusters::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.m_glslCompiler.options();
  options.AddMacroDefinition("CLUSTER_DEDICATED_VERTICES", fmt::format("{}", scene.m_config.clusterDedicatedVertices));
  shaderc::CompileOptions optionsAO = options;
  options.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "0");
  optionsAO.AddMacroDefinition("RAYTRACING_PAYLOAD_INDEX", "1");

  res.compileShader(m_shaders.rayGen, VK_SHADER_STAGE_RAYGEN_BIT_KHR, "render_raytrace.rgen.glsl");
  res.compileShader(m_shaders.rayClosestHit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "render_raytrace_clusters.rchit.glsl");

  res.compileShader(m_shaders.rayMiss, VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl", &options);
  res.compileShader(m_shaders.rayMissAO, VK_SHADER_STAGE_MISS_BIT_KHR, "render_raytrace.rmiss.glsl", &optionsAO);

  res.compileShader(m_shaders.computeBlasInstances, VK_SHADER_STAGE_COMPUTE_BIT, "cluster_blas_instances.comp.glsl");

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
  vkGetPhysicalDeviceProperties2(res.m_physicalDevice, &prop2);

  if(!initRayTracingScene(res, scene, config))
  {
    return false;
  }

  initRayTracingPipeline(res);

  {
    VkPushConstantRange pushRange;
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(shaderio::ClusterBlasConstants);

    VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pPushConstantRanges        = &pushRange;
    layoutInfo.pushConstantRangeCount     = 1;
    vkCreatePipelineLayout(res.m_device, &layoutInfo, nullptr, &m_computePipelineLayout);

    VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
    compInfo.stage.pName                   = "main";
    compInfo.stage.pNext                   = &shaderInfo;
    compInfo.layout                        = m_computePipelineLayout;

    shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_shaders.computeBlasInstances);
    vkCreateComputePipelines(res.m_device, nullptr, 1, &compInfo, nullptr, &m_pipelines.computeBlasInstances);
  }

  return true;
}

void RendererRayTraceClusters::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  vkCmdUpdateBuffer(primary, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants),
                    (const uint32_t*)&frame.frameConstants);
  vkCmdFillBuffer(primary, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);

  if(m_config.doAnimation || m_forceUpdate)
  {
    if(m_config.doAnimation)
    {
      updateAnimation(primary, res, scene, frame, profiler);
    }
    {
      auto timerSection = profiler.cmdFrameSection(primary, "AS Build/Refit");
      updateRayTracingScene(primary, res, scene, frame, profiler);
    }
  }

  VkMemoryBarrier memBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  res.cmdImageTransition(primary, res.m_frameBuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);

  // Ray trace
  {
    auto timerSection = profiler.cmdFrameSection(primary, "Render");

    if(frame.drawObjects)
    {
      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelines.rayTracing);
      vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1,
                              m_dsetPack.sets.data(), 0, nullptr);

      vkCmdTraceRaysKHR(primary, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                        frame.frameConstants.viewport.x, frame.frameConstants.viewport.y, 1);
    }
  }

  {
    // statistics
    shaderio::ClusterBlasConstants blasConstants{};

    // over all clusters
    vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasInstances);

    uint32_t numClusters   = uint32_t(m_clusterSizeBuffer.bufferSize / sizeof(uint32_t));
    blasConstants.sumCount = numClusters;
    blasConstants.sizes    = m_clusterSizeBuffer.address;
    blasConstants.sum      = res.m_commonBuffers.readBack.address + offsetof(shaderio::Readback, clustersSize);

    vkCmdPushConstants(primary, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(shaderio::ClusterBlasConstants), &blasConstants);
    vkCmdDispatch(primary, (numClusters + CLUSTER_BLAS_WORKGROUP_SIZE - 1) / CLUSTER_BLAS_WORKGROUP_SIZE, 1, 1);

    if(!(m_config.doAnimation || m_forceUpdate))
    {
      // get stats for blas

      blasConstants.sumCount = m_blasCount;
      blasConstants.sizes    = m_clusterBlasSizeBuffer.address;
      blasConstants.sum      = res.m_commonBuffers.readBack.address + offsetof(shaderio::Readback, blasesSize);

      vkCmdPushConstants(primary, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(shaderio::ClusterBlasConstants), &blasConstants);
      vkCmdDispatch(primary, (m_blasCount + CLUSTER_BLAS_WORKGROUP_SIZE - 1) / CLUSTER_BLAS_WORKGROUP_SIZE, 1, 1);
    }
  }

  m_forceUpdate = false;
}

void RendererRayTraceClusters::updateRayTracingScene(VkCommandBuffer         cmd,
                                                     Resources&              res,
                                                     Scene&                  scene,
                                                     const FrameConfig&      frame,
                                                     nvvk::ProfilerGpuTimer& profiler)
{
  // wait for animation update
  VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask   = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run template instantiation or clas build
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "clas");
    updateRayTracingClusters(cmd, res, scene);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run blas build
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "blas");
    updateRayTracingBlas(cmd, res, scene);
  }

  memBarrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                             | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier, 0, nullptr, 0, nullptr);


  // fill in per instance blas addresses after the blas were built
  {
    shaderio::ClusterBlasConstants blasConstants{};

    blasConstants.sumCount = m_blasCount;
    blasConstants.sizes    = m_clusterBlasSizeBuffer.address;
    blasConstants.sum      = res.m_commonBuffers.readBack.address + offsetof(shaderio::Readback, blasesSize);

    blasConstants.instanceCount = uint32_t(m_renderInstances.size());
    blasConstants.animated      = m_config.doAnimation ? 1 : 0;
    blasConstants.blasAddresses = m_clusterBlasAddressBuffer.address;
    blasConstants.instances     = m_renderInstanceBuffer.address;
    blasConstants.rayInstances  = m_tlasInstancesBuffer.address;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines.computeBlasInstances);

    vkCmdPushConstants(cmd, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(shaderio::ClusterBlasConstants), &blasConstants);
    vkCmdDispatch(cmd, (std::max(blasConstants.instanceCount, blasConstants.sumCount) + CLUSTER_BLAS_WORKGROUP_SIZE - 1) / CLUSTER_BLAS_WORKGROUP_SIZE,
                  1, 1);
  }

  memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR
                             | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       0, 1, &memBarrier, 0, nullptr, 0, nullptr);

  // run tlas build/update
  {
    auto timerSection = profiler.cmdFrameSection(cmd, "tlas");
    updateRayTracingTlas(cmd, res, scene, !(frame.forceTlasFullRebuild || m_forceUpdate));
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
    cmdInfo.dstAddressesArray.size          = m_clusterDstBuffer.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmdInfo.dstSizesArray.deviceAddress = m_clusterSizeBuffer.address;
    cmdInfo.dstSizesArray.size          = m_clusterSizeBuffer.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = m_instantiationInfoBuffer.address;
    cmdInfo.srcInfosArray.size          = m_instantiationInfoBuffer.bufferSize;
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
    cmdInfo.dstAddressesArray.size          = m_clusterDstBuffer.bufferSize;
    cmdInfo.dstAddressesArray.stride        = sizeof(uint64_t);

    cmdInfo.dstSizesArray.deviceAddress = m_clusterSizeBuffer.address;
    cmdInfo.dstSizesArray.size          = m_clusterSizeBuffer.bufferSize;
    cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

    cmdInfo.srcInfosArray.deviceAddress = m_clusterBuildInfoBuffer.address;
    cmdInfo.srcInfosArray.size          = m_clusterBuildInfoBuffer.bufferSize;
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
  inputs.maxAccelerationStructureCount = m_blasCount;
  inputs.opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
  inputs.opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
  inputs.opInput.pClustersBottomLevel  = &m_clusterBlasInput;
  inputs.flags                         = m_config.clusterBlasFlags;

  // we feed the generated blas addresses directly into the ray instances
  cmdInfo.dstAddressesArray.deviceAddress = m_clusterBlasAddressBuffer.address;
  cmdInfo.dstAddressesArray.size          = m_clusterBlasAddressBuffer.bufferSize;
  cmdInfo.dstAddressesArray.stride        = sizeof(VkDeviceAddress);

  cmdInfo.dstSizesArray.deviceAddress = m_clusterBlasSizeBuffer.address;
  cmdInfo.dstSizesArray.size          = m_clusterBlasSizeBuffer.bufferSize;
  cmdInfo.dstSizesArray.stride        = sizeof(uint32_t);

  cmdInfo.srcInfosArray.deviceAddress = m_clusterBlasInfoBuffer.address;
  cmdInfo.srcInfosArray.size          = m_clusterBlasInfoBuffer.bufferSize;
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
  m_clusterTriangleInput.maxClusterTriangleCount     = scene.m_maxClusterTriangles;
  m_clusterTriangleInput.maxClusterVertexCount       = scene.m_maxClusterVertices;
  m_clusterTriangleInput.maxTotalTriangleCount       = 0;
  m_clusterTriangleInput.maxTotalVertexCount         = 0;
  m_clusterTriangleInput.minPositionTruncateBitCount = config.positionTruncateBits;

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    // in static mode we only build data for the first instance per-geometry
    // which is the one that has the normals
    if(!m_renderInstanceBuffers[i].normals.buffer)
      continue;

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

  res.m_allocator.createBuffer(m_scratchBuffer, m_scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_resourceUsageInfo.rtOtherMemBytes += m_scratchBuffer.bufferSize;

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

  nvvk::Buffer implicitBuffer;

  // we use the same scratch buffer for various operations
  VkDeviceSize tempScratchSize = 0;

  // slightly lower totals because we do one geometry at a time for template builds.
  VkClusterAccelerationStructureTriangleClusterInputNV templateTriangleInput = {
      VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
  templateTriangleInput.vertexFormat                = VK_FORMAT_R32G32B32_SFLOAT;
  templateTriangleInput.maxClusterTriangleCount     = scene.m_maxClusterTriangles;
  templateTriangleInput.maxClusterVertexCount       = scene.m_maxClusterVertices;
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
    res.m_allocator.createBuffer(implicitBuffer, sizesInfo.accelerationStructureSize,
                                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                     | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

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

  nvvk::Buffer scratchBuffer;
  res.m_allocator.createBuffer(scratchBuffer, tempScratchSize,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);

  std::vector<VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV> templateInfos(scene.m_maxPerGeometryClusters);
  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiateInfos(scene.m_maxPerGeometryClusters);

  size_t infoSize = std::max(std::max(sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV),
                                      sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV)),
                             sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV));

  VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  VmaAllocationCreateInfo vmaInfo{};
  vmaInfo.flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  vmaInfo.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  vmaInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;


  nvvk::Buffer infosBuffer;
  bufferInfo.size = infoSize * templateInfos.size();
  res.m_allocator.createBuffer(infosBuffer, bufferInfo, vmaInfo);

  nvvk::Buffer sizesBuffer;
  bufferInfo.size = sizeof(uint32_t) * instantiateInfos.size();
  res.m_allocator.createBuffer(sizesBuffer, bufferInfo, vmaInfo);

  nvvk::Buffer dstAddressesBuffer;
  bufferInfo.size = sizeof(uint64_t) * instantiateInfos.size();
  res.m_allocator.createBuffer(dstAddressesBuffer, bufferInfo, vmaInfo);


  // 32 byte alignment requirement for bbox
  struct TemplateBbox
  {
    shaderio::BBox bbox;
    uint32_t       _pad[2];
  };

  nvvk::Buffer bboxesBuffer;
  res.m_allocator.createBuffer(bboxesBuffer, sizeof(TemplateBbox) * instantiateInfos.size(),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

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
    cmdInfo.srcInfosArray.size              = infosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV);
    cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
    cmdInfo.dstSizesArray.size              = sizesBuffer.bufferSize;
    cmdInfo.dstSizesArray.stride            = sizeof(uint32_t);
    cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
    cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.bufferSize;
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
    res.m_allocator.createBuffer(geometryTemplate.templatesBuffer, buildSum, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV);
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
      cmdInfo.srcInfosArray.size              = infosBuffer.bufferSize;
      cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureMoveObjectsInfoNV);
      cmdInfo.dstSizesArray.deviceAddress     = 0;
      cmdInfo.dstSizesArray.size              = 0;
      cmdInfo.dstSizesArray.stride            = 0;
      cmdInfo.dstAddressesArray.deviceAddress = dstAddressesBuffer.address;
      cmdInfo.dstAddressesArray.size          = dstAddressesBuffer.bufferSize;
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
    cmdInfo.srcInfosArray.size              = infosBuffer.bufferSize;
    cmdInfo.srcInfosArray.stride            = sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV);
    cmdInfo.dstSizesArray.deviceAddress     = sizesBuffer.address;
    cmdInfo.dstSizesArray.size              = sizesBuffer.bufferSize;
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
  res.m_allocator.destroyBuffer(scratchBuffer);
  res.m_allocator.destroyBuffer(infosBuffer);
  res.m_allocator.destroyBuffer(sizesBuffer);
  res.m_allocator.destroyBuffer(dstAddressesBuffer);
  res.m_allocator.destroyBuffer(bboxesBuffer);
  res.m_allocator.destroyBuffer(implicitBuffer);
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
    // in static mode we only build data for the first instance per-geometry
    // which is the one that has the normals
    if(!m_renderInstanceBuffers[i].normals.buffer)
      continue;

    uint32_t geometryID = m_renderInstances[i].geometryID;


    res.m_allocator.createBuffer(m_renderInstanceClusters[i].clusterBuffer, m_geometryTemplates[geometryID].sumInstantionSizes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceUsageInfo.rtClasMemBytes += m_renderInstanceClusters[i].clusterBuffer.bufferSize;

    numClusters += m_geometryTemplates[geometryID].instantionOffsets.size();
  }

  // the actual instantiation process gets argument buffers for the entire scene

  res.m_allocator.createBuffer(m_instantiationInfoBuffer, sizeof(VkClusterAccelerationStructureInstantiateClusterInfoNV) * numClusters,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_instantiationInfoBuffer.bufferSize;

  res.m_allocator.createBuffer(m_clusterDstBuffer, sizeof(uint64_t) * numClusters,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterDstBuffer.bufferSize;

  res.m_allocator.createBuffer(m_clusterSizeBuffer, sizeof(uint32_t) * numClusters,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterSizeBuffer.bufferSize;

  // fill instantiation task data
  // for now on cpu, more typically this would be done on GPU based on culling etc.
  std::vector<VkClusterAccelerationStructureInstantiateClusterInfoNV> instantiationInfos(numClusters, {0});
  std::vector<uint64_t>                                               instantiationDst(numClusters, {0});

  size_t clusterOffset = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    // in static mode we only build data for the first instance per-geometry
    // which is the one that has the normals
    if(!m_renderInstanceBuffers[i].normals.buffer)
      continue;

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

  bool useExplicit = sizesInfo.accelerationStructureSize > std::min(res.m_physicalDeviceInfo.properties11.maxMemoryAllocationSize,
                                                                    res.m_physicalDeviceInfo.properties13.maxBufferSize);

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
      // in static mode we only build data for the first instance per-geometry
      // which is the one that has the normals
      if(!m_renderInstanceBuffers[i].normals.buffer)
        continue;

      uint32_t geometryID = m_renderInstances[i].geometryID;


      res.m_allocator.createBuffer(m_renderInstanceClusters[i].clusterBuffer,
                                   singleExplicitClusterSize * m_renderInstances[i].numClusters,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      m_resourceUsageInfo.rtClasMemBytes += m_renderInstanceClusters[i].clusterBuffer.bufferSize;
    }
  }
  else
  {
    res.m_allocator.createBuffer(m_clusterBuffer, sizesInfo.accelerationStructureSize,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    m_resourceUsageInfo.rtClasMemBytes += m_clusterBuffer.bufferSize;
  }

  // in both cases (explicit and implicit) the argument buffers are for the entire scene

  res.m_allocator.createBuffer(m_clusterBuildInfoBuffer,
                               sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV) * m_numTotalClusters,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterBuildInfoBuffer.bufferSize;

  res.m_allocator.createBuffer(m_clusterDstBuffer, sizeof(uint64_t) * m_numTotalClusters,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterDstBuffer.bufferSize;

  res.m_allocator.createBuffer(m_clusterSizeBuffer, sizeof(uint32_t) * m_numTotalClusters,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterSizeBuffer.bufferSize;

  // fill build info task data
  // for now on cpu, realistically this is done on GPU based on culling etc.
  std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> buildInfos(m_numTotalClusters, {0});
  std::vector<uint64_t>                                                 buildDsts(useExplicit ? m_numTotalClusters : 0);

  size_t clusterOffset = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    // in static mode we only build data for the first instance per-geometry
    // which is the one that has the normals
    if(!m_renderInstanceBuffers[i].normals.buffer)
      continue;

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

  std::vector<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> blasInfos(
      config.doAnimation ? m_renderInstances.size() : scene.m_geometries.size(), {0});

  size_t blasOffset    = 0;
  size_t clusterOffset = 0;
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    // in static mode we only build data for the first instance per-geometry
    // which is the one that has the normals
    if(!m_renderInstanceBuffers[i].normals.buffer)
      continue;

    const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    const Scene::Geometry&          geometry       = scene.m_geometries[renderInstance.geometryID];

    size_t blasOffset = config.doAnimation ? i : renderInstance.geometryID;

    // setup blas/ray instance
    VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV& blasInfo = blasInfos[blasOffset];
    // starting address of array of the dst cluster addresses from instantiation
    // becomes input for blas build
    blasInfo.clusterReferences       = m_clusterDstBuffer.address + sizeof(uint64_t) * clusterOffset;
    blasInfo.clusterReferencesCount  = geometry.numClusters;
    blasInfo.clusterReferencesStride = sizeof(uint64_t);

    clusterOffset += geometry.numClusters;
  }

  // required inputs for blas building and tlas building
  // we will always use implicit mode for this
  res.m_allocator.createBuffer(m_clusterBlasInfoBuffer,
                               sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV) * blasInfos.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterBlasInfoBuffer.bufferSize;

  res.m_allocator.createBuffer(m_clusterBlasSizeBuffer, sizeof(uint32_t) * blasInfos.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterBlasSizeBuffer.bufferSize;

  res.m_allocator.createBuffer(m_clusterBlasAddressBuffer, sizeof(uint64_t) * blasInfos.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += m_clusterBlasAddressBuffer.bufferSize;

  res.simpleUploadBuffer(m_clusterBlasInfoBuffer, blasInfos.data());


  // BLAS space requirement (implicit)
  // the size of the generated blas is dynamic, need to query prebuild info.
  {
    uint32_t blasCount = (uint32_t)blasInfos.size();

    m_clusterBlasInput = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    m_clusterBlasInput.maxClusterCountPerAccelerationStructure = scene.m_maxPerGeometryClusters;
    m_clusterBlasInput.maxTotalClusterCount                    = m_numTotalClusters;

    VkClusterAccelerationStructureInputInfoNV inputs = {VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    inputs.maxAccelerationStructureCount             = blasCount;
    inputs.opMode                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    inputs.opType                       = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    inputs.opInput.pClustersBottomLevel = &m_clusterBlasInput;
    inputs.flags                        = config.clusterBlasFlags;

    VkAccelerationStructureBuildSizesInfoKHR sizesInfo = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(res.m_device, &inputs, &sizesInfo);
    m_scratchSize = std::max(m_scratchSize, sizesInfo.buildScratchSize);

    res.m_allocator.createLargeBuffer(m_clusterBlasBuffer, sizesInfo.accelerationStructureSize,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                      res.m_queue.queue);
    m_resourceUsageInfo.rtBlasMemBytes += m_clusterBlasBuffer.bufferSize;

    m_blasCount = blasCount;
  }

  return true;
}

void RendererRayTraceClusters::initRayTracingPipeline(Resources& res)
{
  VkDevice device = res.m_device;

  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

  m_dsetPack.bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  m_dsetPack.bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  m_dsetPack.bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  m_dsetPack.bindings.addBinding(BINDINGS_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, stageFlags);
  m_dsetPack.bindings.addBinding(BINDINGS_RENDER_TARGET, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stageFlags);
  m_dsetPack.initFromBindings(device, 1);

  nvvk::createPipelineLayout(device, &m_pipelineLayout, {m_dsetPack.layout});

  VkDescriptorImageInfo renderTargetInfo = res.m_frameBuffer.imgColor.descriptor;
  renderTargetInfo.imageLayout           = VK_IMAGE_LAYOUT_GENERAL;

  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_READBACK_SSBO), res.m_commonBuffers.readBack);
  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_TLAS), m_tlas);
  writeSets.append(m_dsetPack.getWriteSet(BINDINGS_RENDER_TARGET), renderTargetInfo);
  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);

  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissAO,
    eClosestHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  std::array<VkShaderModuleCreateInfo, eShaderGroupCount>        stageShaders{};
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stageShaders[s].sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  }
  for(uint32_t s = 0; s < eShaderGroupCount; s++)
  {
    stages[s].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[s].pNext = &stageShaders[s];
    stages[s].pName = "main";
  }

  stages[eRaygen].stage              = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stageShaders[eRaygen].codeSize     = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayGen);
  stageShaders[eRaygen].pCode        = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayGen);
  stages[eMiss].stage                = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMiss].codeSize       = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMiss);
  stageShaders[eMiss].pCode          = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMiss);
  stages[eMissAO].stage              = VK_SHADER_STAGE_MISS_BIT_KHR;
  stageShaders[eMissAO].codeSize     = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayMissAO);
  stageShaders[eMissAO].pCode        = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayMissAO);
  stages[eClosestHit].stage          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stageShaders[eClosestHit].codeSize = nvvkglsl::GlslCompiler::getSpirvSize(m_shaders.rayClosestHit);
  stageShaders[eClosestHit].pCode    = nvvkglsl::GlslCompiler::getSpirv(m_shaders.rayClosestHit);

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                             .generalShader      = VK_SHADER_UNUSED_KHR,
                                             .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                             .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                             .intersectionShader = VK_SHADER_UNUSED_KHR};

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // Miss Ao
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissAO;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  shaderGroups.push_back(group);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount                   = uint32_t(eShaderGroupCount),
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shaderGroups.size()),
      .pGroups                      = shaderGroups.data(),
      .maxPipelineRayRecursionDepth = 2,
      .layout                       = m_pipelineLayout,
  };

  // NEW for clusters! we need to enable their usage explicitly for a ray tracing pipeline
  VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV pipeClusters = {
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
  pipeClusters.allowClusterAccelerationStructure = true;

  rayPipelineInfo.pNext = &pipeClusters;

  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(res.m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_pipelines.rayTracing));
  NVVK_DBG_NAME(m_pipelines.rayTracing);

  // Creating the SBT
  {
    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(res.m_device, m_rtProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_pipelines.rayTracing, rayPipelineInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(res.m_allocator.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
                                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    nvvk::StagingUploader uploader;
    uploader.init(&res.m_allocator);

    void* mapping = nullptr;
    NVVK_CHECK(uploader.appendBufferMapping(m_sbtBuffer, 0, bufferSize, mapping));
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, mapping));

    VkCommandBuffer cmd = res.createTempCmdBuffer();
    uploader.cmdUploadAppended(cmd);
    res.tempSyncSubmit(cmd);
    uploader.deinit();

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}

void RendererRayTraceClusters::deinit(Resources& res)
{
  deinitBasics(res);

  for(auto& it : m_geometryTemplates)
  {
    res.m_allocator.destroyBuffer(it.templatesBuffer);
  }

  for(auto& it : m_renderInstanceClusters)
  {
    res.m_allocator.destroyBuffer(it.clusterBuffer);
  }

  res.m_allocator.destroyBuffer(m_clusterBlasInfoBuffer);
  res.m_allocator.destroyBuffer(m_clusterBlasSizeBuffer);
  res.m_allocator.destroyBuffer(m_clusterBlasAddressBuffer);
  res.m_allocator.destroyBuffer(m_clusterBuffer);
  res.m_allocator.destroyLargeBuffer(m_clusterBlasBuffer);
  res.m_allocator.destroyBuffer(m_clusterDstBuffer);
  res.m_allocator.destroyBuffer(m_clusterSizeBuffer);
  res.m_allocator.destroyBuffer(m_clusterBuildInfoBuffer);
  res.m_allocator.destroyBuffer(m_instantiationInfoBuffer);
  res.m_allocator.destroyBuffer(m_scratchBuffer);
  res.m_allocator.destroyBuffer(m_tlasInstancesBuffer);
  res.m_allocator.destroyBuffer(m_tlasScratchBuffer);
  res.m_allocator.destroyAcceleration(m_tlas);

  res.m_allocator.destroyBuffer(m_sbtBuffer);

  res.destroyPipelines(m_pipelines);

  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);
  vkDestroyPipelineLayout(res.m_device, m_computePipelineLayout, nullptr);

  m_dsetPack.deinit();
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
  renderTargetInfo.imageView   = res.m_frameBuffer.imgColor.descriptor.imageView;
  writeSets[0]                 = m_dsetPack.getWriteSet(BINDINGS_RENDER_TARGET);
  writeSets[0].pImageInfo      = &renderTargetInfo;

  vkUpdateDescriptorSets(res.m_device, uint32_t(writeSets.size()), writeSets.data(), 0, nullptr);
}

}  // namespace animatedclusters
