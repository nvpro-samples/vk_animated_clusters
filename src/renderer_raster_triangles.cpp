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

#include <fmt/format.h>

#include "renderer.hpp"
#include "../shaders/shaderio.h"

namespace animatedclusters {

class RendererRasterTriangles : public Renderer
{
public:
  virtual bool init(Resources& res, Scene& scene, const RendererConfig& config) override;
  virtual void render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler) override;
  virtual void deinit(Resources& res) override;

private:
  bool initShaders(Resources& res, Scene& scene, const RendererConfig& config);

  struct Shaders
  {
    shaderc::SpvCompilationResult trianglesVertex;
    shaderc::SpvCompilationResult trianglesFragment;
    shaderc::SpvCompilationResult backgroundVertex;
    shaderc::SpvCompilationResult backgroundFragment;
  } m_shaders;

  struct Pipelines
  {
    VkPipeline triangles{};
    VkPipeline background{};
  } m_pipelines;

  nvvk::DescriptorPack m_dsetPack;
  VkPipelineLayout     m_pipelineLayout{};
  VkShaderStageFlags   m_stageFlags{};
};

bool RendererRasterTriangles::initShaders(Resources& res, Scene& scene, const RendererConfig& config)
{
  shaderc::CompileOptions options = res.m_glslCompiler.options();
  options.AddMacroDefinition("LINKED_MESH_SHADER", "0");

  res.compileShader(m_shaders.trianglesVertex, VK_SHADER_STAGE_VERTEX_BIT, "render_raster_triangles.vert.glsl", &options);
  res.compileShader(m_shaders.trianglesFragment, VK_SHADER_STAGE_FRAGMENT_BIT, "render_raster.frag.glsl", &options);

  res.compileShader(m_shaders.backgroundVertex, VK_SHADER_STAGE_VERTEX_BIT, "background.vert.glsl");
  res.compileShader(m_shaders.backgroundFragment, VK_SHADER_STAGE_FRAGMENT_BIT, "background.frag.glsl");

  if(!res.verifyShaders(m_shaders))
  {
    return false;
  }

  return initBasicShaders(res, scene, config);
}

bool RendererRasterTriangles::init(Resources& res, Scene& scene, const RendererConfig& config)
{
  VkDevice device = res.m_device;

  m_config = config;

  if(!initShaders(res, scene, config))
    return false;

  initBasics(res, scene, config);

  m_resourceUsageInfo.sceneMemBytes += scene.m_sceneTriangleMemBytes;

  m_stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  nvvk::DescriptorBindings bindings;
  bindings.addBinding(BINDINGS_FRAME_UBO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, m_stageFlags);
  bindings.addBinding(BINDINGS_READBACK_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
  bindings.addBinding(BINDINGS_RENDERINSTANCES_SSBO, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, m_stageFlags);
  m_dsetPack.init(bindings, device, 1);

  nvvk::createPipelineLayout(device, &m_pipelineLayout, {m_dsetPack.getLayout()}, {{m_stageFlags, 0, sizeof(uint32_t)}});

  nvvk::WriteSetContainer writeSets;
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_FRAME_UBO), res.m_commonBuffers.frameConstants);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_READBACK_SSBO), res.m_commonBuffers.readBack);
  writeSets.append(m_dsetPack.makeWrite(BINDINGS_RENDERINSTANCES_SSBO), m_renderInstanceBuffer);
  vkUpdateDescriptorSets(res.m_device, writeSets.size(), writeSets.data(), 0, nullptr);

  {
    nvvk::GraphicsPipelineCreator graphicsGen;
    nvvk::GraphicsPipelineState   graphicsState = res.m_basicGraphicsState;

    graphicsGen.pipelineInfo.layout                  = m_pipelineLayout;
    graphicsGen.renderingState.depthAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.depthAttachmentFormat;
    graphicsGen.renderingState.stencilAttachmentFormat = res.m_frameBuffer.pipelineRenderingInfo.stencilAttachmentFormat;
    graphicsGen.colorFormats = {res.m_frameBuffer.colorFormat};

    graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.trianglesVertex));
    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.trianglesFragment));

    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, graphicsState, &m_pipelines.triangles);

    graphicsGen.clearShaders();

    graphicsState.depthStencilState.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    graphicsGen.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.backgroundVertex));
    graphicsGen.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", nvvkglsl::GlslCompiler::getSpirvData(m_shaders.backgroundFragment));

    graphicsGen.createGraphicsPipeline(res.m_device, nullptr, graphicsState, &m_pipelines.background);
  }
  return true;
}

void RendererRasterTriangles::render(VkCommandBuffer primary, Resources& res, Scene& scene, const FrameConfig& frame, nvvk::ProfilerGpuTimer& profiler)
{
  if(m_config.doAnimation)
  {
    updateAnimation(primary, res, scene, frame, profiler);
  }

  {
    auto timerSection = profiler.cmdFrameSection(primary, "Render");

    vkCmdUpdateBuffer(primary, res.m_commonBuffers.frameConstants.buffer, 0, sizeof(shaderio::FrameConstants),
                      (const uint32_t*)&frame.frameConstants);
    vkCmdFillBuffer(primary, res.m_commonBuffers.readBack.buffer, 0, sizeof(shaderio::Readback), 0);

    nvvk::cmdMemoryBarrier(primary, VK_PIPELINE_STAGE_TRANSFER_BIT, Resources::ALL_SHADER_STAGES);

    bool               useSky = true;
    VkAttachmentLoadOp loadOp = useSky ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR;
    res.cmdBeginRendering(primary, false, loadOp, loadOp);

    vkCmdBindDescriptorSets(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dsetPack.getSets().data(), 0, nullptr);

    if(useSky)
    {
      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.background);
      vkCmdDraw(primary, 3, 1, 0, 0);
    }

    if(frame.drawObjects)
    {
      vkCmdBindPipeline(primary, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelines.triangles);

      for(size_t i = 0; i < m_renderInstances.size(); i++)
      {
        const shaderio::RenderInstance& renderInstance = m_renderInstances[i];
        uint32_t                        instanceIndex  = (uint32_t)i;
        vkCmdPushConstants(primary, m_pipelineLayout, m_stageFlags, 0, sizeof(instanceIndex), &instanceIndex);
        vkCmdBindIndexBuffer(primary, scene.m_geometries[renderInstance.geometryID].trianglesBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(primary, renderInstance.numTriangles * 3, 1, 0, 0, 0);
      }
    }

    vkCmdEndRendering(primary);
  }
}

void RendererRasterTriangles::deinit(Resources& res)
{
  res.destroyPipelines(m_pipelines);
  vkDestroyPipelineLayout(res.m_device, m_pipelineLayout, nullptr);

  m_dsetPack.deinit();

  deinitBasics(res);
}

std::unique_ptr<Renderer> makeRendererRasterTriangles()
{
  return std::make_unique<RendererRasterTriangles>();
}
}  // namespace animatedclusters
