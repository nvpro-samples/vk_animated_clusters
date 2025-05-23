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

#include "renderer.hpp"
#include "shaders/shaderio.h"

namespace animatedclusters {

class RendererRasterTriangles : public Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  struct Shaders
  {
    nvvk::ShaderModuleID vertexShader;
    nvvk::ShaderModuleID fragmentShader;
  } m_shaders;


  nvvk::DescriptorSetContainer m_dsetContainer;
  VkPipeline                   m_pipeline = nullptr;
};

bool RendererRasterTriangles::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_shaders.vertexShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "render_raster_triangles.vert.glsl");
  m_shaders.fragmentShader = res.m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl");
  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRasterTriangles::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  m_config = config;

  if(!initShaders(res, scene, config))
    return false;

  initBasics(res, scene, config);

  m_resourceUsageInfo.sceneMemBytes += scene.m_sceneTriangleMemBytes;

  m_dsetContainer.init(res.m_device);

  VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  m_dsetContainer.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stageFlags);
  m_dsetContainer.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  m_dsetContainer.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stageFlags);
  m_dsetContainer.initLayout();

  VkPushConstantRange pushRange;
  pushRange.offset     = 0;
  pushRange.size       = sizeof(uint32_t);
  pushRange.stageFlags = stageFlags;
  m_dsetContainer.initPipeLayout(1, &pushRange);

  m_dsetContainer.initPool(1);
  VkWriteDescriptorSet writeSets[3];
  writeSets[0] = m_dsetContainer.makeWrite(0, BINDINGS_FRAME_UBO, &res.m_common.view.info);
  writeSets[1] = m_dsetContainer.makeWrite(0, BINDINGS_READBACK_SSBO, &res.m_common.readbackDevice.info);
  writeSets[2] = m_dsetContainer.makeWrite(0, BINDINGS_RENDERINSTANCES_SSBO, &m_renderInstanceBuffer.info);
  vkUpdateDescriptorSets(res.m_device, 3, writeSets, 0, nullptr);

  nvvk::GraphicsPipelineState     state = res.m_basicGraphicsState;
  nvvk::GraphicsPipelineGenerator gfxGen(res.m_device, m_dsetContainer.getPipeLayout(),
                                         res.m_framebuffer.pipelineRenderingInfo, state);
  gfxGen.addShader(res.m_shaderManager.get(m_shaders.vertexShader), VK_SHADER_STAGE_VERTEX_BIT);
  gfxGen.addShader(res.m_shaderManager.get(m_shaders.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pipeline = gfxGen.createPipeline();

  return true;
}

void RendererRasterTriangles::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerVK& profiler)
{
  if(m_config.doAnimation)
  {
    updateAnimation(primary, res, scene, frame, profiler);
  }

  const bool useSky = true;  // When using Sky, the sky is rendered first and the rest of the scene is rendered on top of it.

  {
    auto timerSection = profiler.timeRecurring("Render", primary);

    vkCmdUpdateBuffer(primary, res.m_common.view.buffer, 0, sizeof(shaderio::FrameConstants), (const uint32_t*)&frame.frameConstants);
    vkCmdFillBuffer(primary, res.m_common.readbackDevice.buffer, 0, sizeof(shaderio::Readback), 0);
    
    if(useSky)
    {
      res.m_sky.skyParams() = frame.frameConstants.skyParams;
      res.m_sky.updateParameterBuffer(primary);
      res.cmdImageTransition(primary, res.m_framebuffer.imgColor, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL);
      res.m_sky.draw(primary, frame.frameConstants.viewMatrix, frame.frameConstants.projMatrix,
                     res.m_framebuffer.scissor.extent);
    }

    res.cmdBeginRendering(primary, false, useSky ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR);

    if(frame.drawObjects)
    {
      res.cmdDynamicState(primary);
      vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dsetContainer.getPipeLayout(), 0, 1,
                              m_dsetContainer.getSets(), 0, nullptr);
      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

      for(size_t i = 0; i < m_renderInstances.size(); i++)
      {
        const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
        uint32_t                        instanceIndex  = (uint32_t)i;
        vkCmdPushConstants(primary, m_dsetContainer.getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(instanceIndex), &instanceIndex);
        vkCmdBindIndexBuffer(primary, scene.m_geometries[renderInstance.geometryID].trianglesBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(primary, renderInstance.numTriangles * 3, 1, 0, 0, 0);
      }
    }

    vkCmdEndRendering(primary);
  }
}

void RendererRasterTriangles::deinit(Resources& res)
{
  vkDestroyPipeline(res.m_device, m_pipeline, nullptr);

  m_dsetContainer.deinit();

  res.destroyShaders(m_shaders);

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterTriangles()
{
  return std::make_unique<RendererRasterTriangles>();
}
}  // namespace animatedclusters
