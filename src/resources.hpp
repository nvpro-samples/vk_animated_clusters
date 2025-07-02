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

#include <span>

#include <glm/glm.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvkglsl/glsl.hpp>

#include "hbao_pass.hpp"
#include "../shaders/shaderio.h"

namespace animatedclusters {

struct FrameConfig
{
  VkExtent2D               windowSize;
  bool                     hbaoActive       = true;
  bool                     drawClusterBoxes = false;
  bool                     drawObjects      = true;
  uint32_t                 rebuildNth;
  bool                     forceTlasFullRebuild = false;
  float                    blasRebuildFraction  = 0.1f;
  shaderio::FrameConstants frameConstants;
  HbaoPass::Settings       hbaoSettings;
};

//////////////////////////////////////////////////////////////////////////

struct PhysicalDeviceInfo
{
  VkPhysicalDeviceProperties         properties10;
  VkPhysicalDeviceVulkan11Properties properties11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
  VkPhysicalDeviceVulkan12Properties properties12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
  VkPhysicalDeviceVulkan13Properties properties13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
  VkPhysicalDeviceVulkan14Properties properties14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES};

  VkPhysicalDeviceFeatures         features10;
  VkPhysicalDeviceVulkan11Features features11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan14Features features14 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};

  void init(VkPhysicalDevice physicalDevice, uint32_t apiVersion = VK_API_VERSION_1_4)
  {
    assert(apiVersion >= VK_API_VERSION_1_2);

    VkPhysicalDeviceProperties2 props = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext                       = &properties11;
    properties11.pNext                = &properties12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      properties12.pNext = &properties13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      properties13.pNext = &properties14;
    }
    vkGetPhysicalDeviceProperties2(physicalDevice, &props);
    properties10 = props.properties;

    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext                     = &features11;
    features11.pNext                   = &features12;
    if(apiVersion >= VK_API_VERSION_1_3)
    {
      features12.pNext = &features13;
    }
    if(apiVersion >= VK_API_VERSION_1_4)
    {
      features13.pNext = &features14;
    }
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features);
    features10 = features.features;
  }
};

//////////////////////////////////////////////////////////////////////////

class Resources
{
public:
  static constexpr VkPipelineStageFlags2 ALL_SHADER_STAGES =
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;


  struct FrameBuffer
  {
    VkExtent2D renderSize{};

    int  supersample = 0;
    bool useResolved = false;

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depthStencilFormat;

    VkViewport viewport;
    VkRect2D   scissor;

    nvvk::Image imgColor         = {};
    nvvk::Image imgColorResolved = {};
    nvvk::Image imgDepthStencil  = {};

    VkImageView viewDepth = VK_NULL_HANDLE;

    VkPipelineRenderingCreateInfo pipelineRenderingInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  };


  void init(VkDevice device, VkPhysicalDevice physicalDevice, VkInstance instance, nvvk::QueueInfo queue);
  void deinit();

  bool initFramebuffer(const VkExtent2D& windowSize, int supersample, bool hbaoFullRes);
  void deinitFramebuffer();

  void beginFrame(uint32_t cycleIndex);
  void blitFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void emptyFrame(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);
  void endFrame();

  void cmdHBAO(VkCommandBuffer cmd, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler);

  void getReadbackData(shaderio::Readback& readback);

  //////////////////////////////////////////////////////////////////////////

  shaderc::CompileOptions makeCompilerOptions() { return shaderc::CompileOptions(m_glslCompiler.options()); }

  bool compileShader(shaderc::SpvCompilationResult& compiled,
                     VkShaderStageFlagBits          shader,
                     const std::filesystem::path&   filePath,
                     shaderc::CompileOptions*       options = nullptr);

  // tests if all shaders compiled well, returns false if not
  // also destroys all shaders if not all were successful.
  bool verifyShaders(size_t numShaders, shaderc::SpvCompilationResult* shaders)
  {
    for(size_t i = 0; i < numShaders; i++)
    {
      if(shaders[i].GetCompilationStatus() != shaderc_compilation_status_null_result_object
         && shaders[i].GetCompilationStatus() != shaderc_compilation_status_success)
        return false;
    }

    return true;
  }
  template <typename T>
  bool verifyShaders(T& container)
  {
    return verifyShaders(sizeof(T) / sizeof(shaderc::SpvCompilationResult), (shaderc::SpvCompilationResult*)&container);
  }

  void destroyPipelines(size_t numPipelines, VkPipeline* pipelines)
  {
    for(size_t i = 0; i < numPipelines; i++)
    {
      vkDestroyPipeline(m_device, pipelines[i], nullptr);
      pipelines[i] = nullptr;
    }
  }
  template <typename T>
  void destroyPipelines(T& container)
  {
    destroyPipelines(sizeof(T) / sizeof(VkPipeline), (VkPipeline*)&container);
  }

  //////////////////////////////////////////////////////////////////////////

  VkCommandBuffer createTempCmdBuffer();
  void            tempSyncSubmit(VkCommandBuffer cmd);

  //////////////////////////////////////////////////////////////////////////

  void cmdBeginRendering(VkCommandBuffer    cmd,
                         bool               hasSecondary = false,
                         VkAttachmentLoadOp loadOpColor  = VK_ATTACHMENT_LOAD_OP_CLEAR,
                         VkAttachmentLoadOp loadOpDepth  = VK_ATTACHMENT_LOAD_OP_CLEAR);
  void cmdBeginRayTracing(VkCommandBuffer cmd);

  void cmdImageTransition(VkCommandBuffer cmd, nvvk::Image& rimg, VkImageAspectFlags aspects, VkImageLayout newLayout, bool needBarrier = false) const;

  //////////////////////////////////////////////////////////////////////////

  void simpleUploadBuffer(const nvvk::Buffer& buffer, void* data)
  {
    VkCommandBuffer cmd = createTempCmdBuffer();
    m_uploader.appendBuffer(buffer, 0, buffer.bufferSize, data);
    m_uploader.cmdUploadAppended(cmd);
    tempSyncSubmit(cmd);
    m_uploader.releaseStaging();
  }

  enum FlushState
  {
    ALLOW_FLUSH,
    DONT_FLUSH,
  };

  class BatchedUploader
  {
  public:
    BatchedUploader(Resources& resources, VkDeviceSize maxBatchSize = 128 * 1024 * 1024)
        : m_resources(resources)
        , m_maxBatchSize(maxBatchSize)
    {
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, size_t offset, size_t sz, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      if(sz)
      {
        if(m_resources.m_uploader.checkAppendedSize(m_maxBatchSize, sz) && flushState == FlushState::ALLOW_FLUSH)
        {
          flush();
        }

        if(!m_cmd)
        {
          m_cmd = m_resources.createTempCmdBuffer();
        }
        T* mapping = nullptr;
        NVVK_CHECK(m_resources.m_uploader.appendBufferMapping(dst, offset, sz, mapping));

        if(src)
        {
          memcpy(mapping, src, sz);
        }

        return mapping;
      }
      return nullptr;
    }

    template <typename T>
    T* uploadBuffer(const nvvk::Buffer& dst, const T* src, FlushState flushState = FlushState::ALLOW_FLUSH)
    {
      return uploadBuffer(dst, 0, dst.bufferSize, src, flushState);
    }

    void fillBuffer(const nvvk::Buffer& dst, uint32_t fillValue)
    {
      if(!m_cmd)
      {
        m_cmd = m_resources.createTempCmdBuffer();
      }
      vkCmdFillBuffer(m_cmd, dst.buffer, 0, dst.bufferSize, fillValue);
    }

    // must call flush at end of operations
    void flush()
    {
      if(m_cmd)
      {
        m_resources.m_uploader.cmdUploadAppended(m_cmd);
        m_resources.tempSyncSubmit(m_cmd);
        m_resources.m_uploader.releaseStaging();
        m_cmd = nullptr;
      }
    }

    ~BatchedUploader() { assert(!m_cmd && "must call flush at end"); }

  private:
    Resources&      m_resources;
    VkDeviceSize    m_maxBatchSize = 0;
    VkCommandBuffer m_cmd          = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////

  static constexpr VkPipelineStageFlags2 s_supportedShaderStages =
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT
      | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

  VkDevice         m_device          = {};
  VkPhysicalDevice m_physicalDevice  = {};
  nvvk::QueueInfo  m_queue           = {};
  VkCommandPool    m_tempCommandPool = {};

  nvvk::ResourceAllocator m_allocator    = {};
  nvvk::SamplerPool       m_samplerPool  = {};
  nvvkglsl::GlslCompiler  m_glslCompiler = {};
  nvvk::StagingUploader   m_uploader     = {};

  FrameBuffer m_frameBuffer;
  struct CommonBuffers
  {
    nvvk::BufferTyped<shaderio::FrameConstants> frameConstants;
    nvvk::BufferTyped<shaderio::Readback>       readBack;
    nvvk::BufferTyped<shaderio::Readback>       readBackHost;
  } m_commonBuffers;

  PhysicalDeviceInfo          m_physicalDeviceInfo = {};
  nvvk::GraphicsPipelineState m_basicGraphicsState = {};
  uint32_t                    m_cycleIndex         = 0;
  size_t                      m_fboChangeID        = ~0;
  glm::vec4                   m_bgColor            = {0.1, 0.13, 0.15, 1.0};
  bool                        m_supportsClusters   = false;

  bool            m_hbaoFullRes = false;
  HbaoPass        m_hbaoPass;
  HbaoPass::Frame m_hbaoFrame;

private:
};


}  // namespace animatedclusters
