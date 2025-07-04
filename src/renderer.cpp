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

#include <random>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "renderer.hpp"
#include "../shaders/shaderio.h"

namespace animatedclusters {

bool Renderer::initBasicShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  res.compileShader(m_basicShaders.animComputeVertices, VK_SHADER_STAGE_COMPUTE_BIT, "animupdate_vertices.comp.glsl");
  res.compileShader(m_basicShaders.animComputeNormals, VK_SHADER_STAGE_COMPUTE_BIT, "animupdate_normals.comp.glsl");

  if(!res.verifyShaders(m_basicShaders))
  {
    return false;
  }

  return true;
}

void Renderer::initBasics(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_resourceUsageInfo = {};

  nvvk::createPipelineLayout(res.m_device, &m_animPipelineLayout, {},
                             {{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::AnimationConstants)}});

  VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  VkComputePipelineCreateInfo info       = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  info.layout                            = m_animPipelineLayout;
  info.stage                             = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  info.stage.stage                       = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.pName                       = "main";
  info.stage.pNext                       = &shaderInfo;

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_basicShaders.animComputeVertices);
  vkCreateComputePipelines(res.m_device, nullptr, 1, &info, nullptr, &m_basicPipelines.animComputeVertices);

  shaderInfo = nvvkglsl::GlslCompiler::makeShaderModuleCreateInfo(m_basicShaders.animComputeNormals);
  vkCreateComputePipelines(res.m_device, nullptr, 1, &info, nullptr, &m_basicPipelines.animComputeNormals);

  m_renderInstances.resize(scene.m_instances.size() * config.numSceneCopies);
  m_renderInstanceBuffers.resize(m_renderInstances.size());

  std::default_random_engine            rng(2342);
  std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

  uint32_t axis    = config.gridConfig;
  size_t   sq      = 1;
  int      numAxis = 0;
  if(!axis)
    axis = 3;

  for(int i = 0; i < 3; i++)
  {
    numAxis += (axis & (1 << i)) ? 1 : 0;
  }

  switch(numAxis)
  {
    case 1:
      sq = config.numSceneCopies;
      break;
    case 2:
      while(sq * sq < config.numSceneCopies)
      {
        sq++;
      }
      break;
    case 3:
      while(sq * sq * sq < config.numSceneCopies)
      {
        sq++;
      }
      break;
  }

  VkCommandBuffer cmd = res.createTempCmdBuffer();

  size_t lastCopyIndex = 0;

  glm::vec3 gridShift;
  glm::mat4 gridRotMatrix;

  m_geometryFirstInstance.resize(scene.m_geometries.size(), ~0);

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    size_t originalIndex = i % scene.m_instances.size();
    size_t copyIndex     = i / scene.m_instances.size();

    shaderio::RenderInstance& renderInstance = m_renderInstances[i];
    renderInstance                           = {};

    const uint32_t         geometryID = scene.m_instances[originalIndex].geometryID;
    const Scene::Geometry& geometry   = scene.m_geometries[geometryID];

    glm::mat4 worldMatrix = scene.m_instances[originalIndex].matrix;

    bool isFirstInstance = false;
    if(m_geometryFirstInstance[geometryID] == ~0)
    {
      m_geometryFirstInstance[geometryID] = uint32_t(i);
      isFirstInstance                     = true;
    }

    if(copyIndex)
    {
      if(copyIndex != lastCopyIndex)
      {
        lastCopyIndex = copyIndex;

        gridShift = config.refShift * (scene.m_bbox.hi - scene.m_bbox.lo);
        size_t c  = copyIndex;

        float u = 0;
        float v = 0;
        float w = 0;

        switch(numAxis)
        {
          case 1:
            u = float(c);
            break;
          case 2:
            u = float(c % sq);
            v = float(c / sq);
            break;
          case 3:
            u = float(c % sq);
            v = float((c / sq) % sq);
            w = float(c / (sq * sq));
            break;
        }

        float use = u;

        if(axis & (1 << 0))
        {
          gridShift.x *= -use;
          if(numAxis > 1)
            use = v;
        }
        else
        {
          gridShift.x = 0;
        }

        if(axis & (1 << 1))
        {
          gridShift.y *= use;
          if(numAxis > 2)
            use = w;
          else if(numAxis > 1)
            use = v;
        }
        else
        {
          gridShift.y = 0;
        }

        if(axis & (1 << 2))
        {
          gridShift.z *= -use;
        }
        else
        {
          gridShift.z = 0;
        }

        if(axis & (8 | 16 | 32))
        {
          glm::vec3 mask    = {axis & 8 ? 1.0f : 0.0f, axis & 16 ? 1.0f : 0.0f, axis & 32 ? 1.0f : 0.0f};
          glm::vec3 gridDir = glm::vec3(randomUnorm(rng), randomUnorm(rng), randomUnorm(rng));
          gridDir           = glm::max(gridDir * mask, mask * 0.00001f);
          float gridAngle   = randomUnorm(rng) * glm::pi<float>() * 2.0f;
          gridDir           = glm::normalize(gridDir);

          gridRotMatrix = glm::rotate(glm::mat4(1), gridAngle, gridDir);
        }
      }

      glm::vec3 translation;
      translation = worldMatrix[3];
      if(axis & (8 | 16 | 32))
      {
        worldMatrix[3] = glm::vec4(0, 0, 0, 1);
        worldMatrix    = gridRotMatrix * worldMatrix;
      }
      worldMatrix[3] = glm::vec4(translation + gridShift, 1.f);
    }

    renderInstance.worldMatrix  = worldMatrix;
    renderInstance.numVertices  = geometry.numVertices;
    renderInstance.numClusters  = geometry.numClusters;
    renderInstance.numTriangles = geometry.numTriangles;
    renderInstance.geometryID   = geometryID;

    // original data
    renderInstance.triangles             = geometry.trianglesBuffer.address;
    renderInstance.clusters              = geometry.clustersBuffer.address;
    renderInstance.clusterLocalTriangles = geometry.clusterLocalTrianglesBuffer.address;
    renderInstance.clusterLocalVertices  = geometry.clusterLocalVerticesBuffer.address;
    renderInstance.clusterBboxes         = geometry.clusterBboxesBuffer.address;
    renderInstance.originalPositions     = geometry.positionsBuffer.address;


    RenderInstanceData& renderInstanceData = m_renderInstanceBuffers[i];

    if(config.doAnimation)
    {
      // animated, each instance gets its own copy

      res.m_allocator.createBuffer(renderInstanceData.positions, sizeof(glm::vec3) * geometry.numVertices,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

      res.m_allocator.createBuffer(renderInstanceData.normals, sizeof(glm::vec3) * geometry.numVertices,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

      m_resourceUsageInfo.sceneMemBytes += renderInstanceData.positions.bufferSize;
      m_resourceUsageInfo.sceneMemBytes += renderInstanceData.normals.bufferSize;

      // seed with original data
      VkBufferCopy region;
      region.dstOffset = 0;
      region.size      = sizeof(glm::vec3) * geometry.numVertices;
      region.srcOffset = 0;
      vkCmdCopyBuffer(cmd, geometry.positionsBuffer.buffer, renderInstanceData.positions.buffer, 1, &region);

      vkCmdFillBuffer(cmd, renderInstanceData.normals.buffer, 0, renderInstanceData.normals.bufferSize, 0);

      renderInstance.positions = renderInstanceData.positions.address;
      renderInstance.normals   = renderInstanceData.normals.address;
    }
    else if(isFirstInstance)
    {
      // first instance for a geometry will allocate a normal buffer, that all other instances will share

      res.m_allocator.createBuffer(renderInstanceData.normals, sizeof(glm::vec3) * geometry.numVertices,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

      m_resourceUsageInfo.sceneMemBytes += renderInstanceData.normals.bufferSize;

      renderInstance.positions = geometry.positionsBuffer.address;
      renderInstance.normals   = renderInstanceData.normals.address;
    }
    else
    {
      renderInstance.positions = geometry.positionsBuffer.address;
      renderInstance.normals   = m_renderInstanceBuffers[m_geometryFirstInstance[geometryID]].normals.address;
    }
  }
  res.tempSyncSubmit(cmd);
  res.m_allocator.createBuffer(m_renderInstanceBuffer, sizeof(shaderio::RenderInstance) * m_renderInstances.size(),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  res.simpleUploadBuffer(m_renderInstanceBuffer, m_renderInstances.data());

  m_resourceUsageInfo.sceneMemBytes += m_renderInstanceBuffer.bufferSize;

  {
    cmd = res.createTempCmdBuffer();

    // update normals once
    shaderio::AnimationConstants constants;
    constants.animationState = 0;
    constants.instances      = m_renderInstanceBuffer.address;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_basicPipelines.animComputeNormals);

    for(size_t i = 0; i < m_renderInstances.size(); i++)
    {
      // without animation, only a subset of instances would have unique normals

      if(!m_renderInstanceBuffers[i].normals.buffer)
        continue;

      constants.instanceIndex = uint32_t(i);
      vkCmdPushConstants(cmd, m_animPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
      vkCmdDispatch(cmd, nvvk::getGroupCounts(m_renderInstances[i].numTriangles, ANIMATION_WORKGROUP_SIZE), 1, 1);
    }

    res.tempSyncSubmit(cmd);
  }
}


void Renderer::deinitBasics(Resources& res)
{
  res.destroyPipelines(m_basicPipelines);

  vkDestroyPipelineLayout(res.m_device, m_animPipelineLayout, nullptr);

  for(auto& it : m_renderInstanceBuffers)
  {
    res.m_allocator.destroyBuffer(it.positions);
    res.m_allocator.destroyBuffer(it.normals);
  }

  res.m_allocator.destroyBuffer(m_renderInstanceBuffer);
}

void Renderer::updateAnimation(VkCommandBuffer cmd, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  assert(m_config.doAnimation);

  auto timerSection = profiler.cmdFrameSection(cmd, "Animation");

  shaderio::AnimationConstants constants;
  constants.animationState = frame.frameConstants.animationState;
  constants.instances      = m_renderInstanceBuffer.address;

  constants.rippleEnabled   = frame.frameConstants.animationRippleEnabled;
  constants.rippleFrequency = frame.frameConstants.animationRippleFrequency;
  constants.rippleAmplitude = frame.frameConstants.animationRippleAmplitude;
  constants.rippleSpeed     = frame.frameConstants.animationRippleSpeed;
  constants.twistEnabled    = frame.frameConstants.animationTwistEnabled;
  constants.twistSpeed      = frame.frameConstants.animationTwistSpeed;
  constants.twistMaxAngle   = frame.frameConstants.animationTwistMaxAngle * glm::pi<float>() / 180.f;

  // first pass all positions
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_basicPipelines.animComputeVertices);
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    const Scene::Geometry& sceneGeometry = scene.m_geometries[m_renderInstances[i].geometryID];

    constants.instanceIndex = uint32_t(i);

    // offset by instance
    constants.animationState = frame.frameConstants.animationState + float(i);

    constants.geometrySize = glm::length(sceneGeometry.bbox.hi - sceneGeometry.bbox.lo);

    vkCmdPushConstants(cmd, m_animPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(cmd, nvvk::getGroupCounts(m_renderInstances[i].numVertices, ANIMATION_WORKGROUP_SIZE), 1, 1);
  }

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

  // second pass all normals
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_basicPipelines.animComputeNormals);
  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    constants.instanceIndex = uint32_t(i);
    vkCmdPushConstants(cmd, m_animPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    vkCmdDispatch(cmd, nvvk::getGroupCounts(m_renderInstances[i].numTriangles, ANIMATION_WORKGROUP_SIZE), 1, 1);
  }
}

void Renderer::initRayTracingTlas(Resources& res, Scene& scene, const RendererConfig& config, const VkAccelerationStructureKHR* blas)
{
  std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(m_renderInstances.size());

  for(size_t i = 0; i < m_renderInstances.size(); i++)
  {
    VkDeviceAddress blasAddress{};
    if(blas != nullptr)
    {
      VkAccelerationStructureDeviceAddressInfoKHR addressInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
      addressInfo.accelerationStructure = blas[i];
      blasAddress                       = vkGetAccelerationStructureDeviceAddressKHR(res.m_device, &addressInfo);
    }

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform                              = nvvk::toTransformMatrixKHR(m_renderInstances[i].worldMatrix);
    instance.instanceCustomIndex                    = static_cast<uint32_t>(i);  // gl_InstanceCustomIndexEX
    instance.mask                                   = 0xFF;                      // All objects
    instance.instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
        instance.flags                              = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    instance.accelerationStructureReference         = blasAddress;
    tlasInstances[i]                                = instance;
  }


  // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
  res.m_allocator.createBuffer(m_tlasInstancesBuffer, tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_resourceUsageInfo.rtOtherMemBytes += tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);
  res.simpleUploadBuffer(m_tlasInstancesBuffer, tlasInstances.data());

  // Wraps a device pointer to the above uploaded instances.
  VkAccelerationStructureGeometryInstancesDataKHR instancesVk{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  instancesVk.data.deviceAddress = m_tlasInstancesBuffer.address;

  // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label it as instance data.
  m_tlasGeometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  m_tlasGeometry.geometryType       = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  m_tlasGeometry.geometry.instances = instancesVk;

  // Find sizes
  m_tlasBuildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  m_tlasBuildInfo.flags         = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_tlasBuildInfo.geometryCount = 1;
  m_tlasBuildInfo.pGeometries   = &m_tlasGeometry;
  // FIXME
  m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  m_tlasBuildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  uint32_t                                 instanceCount = uint32_t(m_renderInstances.size());
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(res.m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                          &m_tlasBuildInfo, &instanceCount, &sizeInfo);


  // Create TLAS
  VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  createInfo.size = sizeInfo.accelerationStructureSize;

  res.m_allocator.createAcceleration(m_tlas, createInfo);
  m_resourceUsageInfo.rtTlasMemBytes += createInfo.size;
  // Allocate the scratch memory
  res.m_allocator.createBuffer(m_tlasScratchBuffer, sizeInfo.buildScratchSize,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
  m_resourceUsageInfo.rtOtherMemBytes += sizeInfo.buildScratchSize;

  // Update build information
  m_tlasBuildInfo.srcAccelerationStructure  = VK_NULL_HANDLE;
  m_tlasBuildInfo.dstAccelerationStructure  = m_tlas.accel;
  m_tlasBuildInfo.scratchData.deviceAddress = m_tlasScratchBuffer.address;
}

void Renderer::updateRayTracingTlas(VkCommandBuffer cmd, Resources& res, Scene& scene, bool update)
{
  if(update)
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = m_tlas.accel;
  }
  else
  {
    m_tlasBuildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    m_tlasBuildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
  }

  // Build Offsets info: n instances
  VkAccelerationStructureBuildRangeInfoKHR        buildOffsetInfo{uint32_t(m_renderInstances.size()), 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffsetInfo = &buildOffsetInfo;

  // Build the TLAS
  vkCmdBuildAccelerationStructuresKHR(cmd, 1, &m_tlasBuildInfo, &pBuildOffsetInfo);
}


}  // namespace animatedclusters
