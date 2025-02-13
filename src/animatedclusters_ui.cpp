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

#include <filesystem>

#include <imgui/backends/imgui_vk_extra.h>
#include <imgui/imgui_camera_widget.h>
#include <imgui/imgui_orient.h>
#include <implot.h>
#include <nvh/fileoperations.hpp>
#include <nvh/cameramanipulator.hpp>
#include <nvh/misc.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvkhl/sky.hpp>

#include "animatedclusters.hpp"
#include "vk_nv_cluster_acc.h"

namespace animatedclusters {

std::string formatMemorySize(size_t sizeInBytes)
{
  static const std::string units[]     = {"B", "KB", "MB", "GB"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(sizeInBytes < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float size = float(sizeInBytes) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", size, units[currentUnit]);
}

std::string formatMetric(size_t size)
{
  static const std::string units[]     = {"", "K", "M", "G"};
  static const size_t      unitSizes[] = {1, 1000, 1000 * 1000, 1000 * 1000 * 1000};

  uint32_t currentUnit = 0;
  for(uint32_t i = 1; i < 4; i++)
  {
    if(size < unitSizes[i])
    {
      break;
    }
    currentUnit++;
  }

  float fsize = float(size) / float(unitSizes[currentUnit]);

  return fmt::format("{:.3} {}", fsize, units[currentUnit]);
}

void AnimatedClusters::processUI(double time, nvh::Profiler& profiler, const CallBacks& callbacks)
{
  bool earlyOut = !m_scene;

  if(earlyOut)
  {
    return;
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  bool isRightClick = m_mouseButtonHandler.getButtonState(ImGuiMouseButton_Right) == MouseButtonHandler::eSingleClick;

  //TODO(JEM) the readback meshanism for picking is not fully functional yet, see with MKL
  // In the current status of implementation the isReadbackValid is allways false
  //
  // camera control, recenter
  if(m_requestCameraRecenter && isReadbackValid(readback))
  {

    glm::uvec2 mousePos = {m_frameConfig.frameConstants.mousePosition.x / m_tweak.supersample,
                           m_frameConfig.frameConstants.mousePosition.y / m_tweak.supersample};

    const glm::mat4 view = CameraManip.getMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4       win_norm = {0, 0, m_frameConfig.frameConstants.viewport.x / m_tweak.supersample,
                                  m_frameConfig.frameConstants.viewport.y / m_tweak.supersample};
      const glm::vec3 hitPos   = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      // Set the interest position
      glm::vec3 eye, center, up;
      CameraManip.getLookat(eye, center, up);
      CameraManip.setLookat(eye, hitPos, up, false);
    }
  }

  // for emphasized parameter we want to recommend to the user
  const ImVec4 recommendedColor = ImVec4(0.0, 1.0, 0.0, 1.0);

  namespace PE = ImGuiH::PropertyEditor;

  ImGui::Begin("Settings");
  ImGui::PushItemWidth(ImGuiH::dpiScaled(170));

  if(ImGui::CollapsingHeader("Scene Modifiers", nullptr))  // default closed
  {
    PE::begin("##scene");
    PE::InputInt("Number of scene copies", (int*)&m_tweak.gridCopies, 1, 16, ImGuiInputTextFlags_EnterReturnsTrue,
                 "replicates the gltf model on a grid layout");
    PE::InputInt("Layout grid axis bits", (int*)&m_tweak.gridConfig, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue,
                 "layout grid config encoded in 6 bits: 0..2 bit enabled axis, 3..5 bit enabled rotation");
    PE::end();
  }
  if(ImGui::CollapsingHeader("Rendering", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Rendering");
    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::entry("Renderer", [&]() {
      ImGui::PopStyleColor();  // pop text color here so it only applies to the label
      return m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer);
    });

    PE::entry("Super sampling", [&]() { return m_ui.enumCombobox(GUI_SUPERSAMPLE, "sampling", &m_tweak.supersample); });
    PE::Text("Render Resolution:", "%d x %d", m_resources.m_framebuffer.renderWidth, m_resources.m_framebuffer.renderHeight);
    PE::Checkbox("Facet shading", &m_tweak.facetShading);

    // conditional UI, declutters the UI, prevents presenting many sections in disabled state
    if(m_tweak.renderer == RENDERER_RAYTRACE_TRIANGLES || m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS)
    {
      PE::Checkbox("Cast shadow rays", (bool*)&m_frameConfig.frameConstants.doShadow);
      PE::SliderFloat("Ambient occlusion radius", &m_frameConfig.frameConstants.ambientOcclusionRadius, 0.001f, 1.f);
      PE::SliderInt("Ambient occlusion rays", &m_frameConfig.frameConstants.ambientOcclusionRays, 0, 64);
    }
    if(m_tweak.renderer == RENDERER_RASTER_TRIANGLES || m_tweak.renderer == RENDERER_RASTER_CLUSTERS)
    {
      PE::Checkbox("Ambient occlusion (HBAO)", &m_tweak.hbaoActive);
      if(PE::treeNode("HBAO settings"))
      {
        PE::Checkbox("Full resolution", &m_tweak.hbaoFullRes);
        PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f);
        PE::InputFloat("Blur sharpness", &m_frameConfig.hbaoSettings.blurSharpness, 1.0f);
        PE::InputFloat("Intensity", &m_frameConfig.hbaoSettings.intensity, 0.1f);
        PE::InputFloat("Bias", &m_frameConfig.hbaoSettings.bias, 0.01f);
        PE::treePop();
      }
    }
    PE::end();
    ImGui::TextDisabled("Clusters specifics");
    PE::begin("##RenderingSpecifics");
    ImGui::PushStyleColor(ImGuiCol_Text, recommendedColor);
    PE::entry("Visualize", [&]() {
      ImGui::PopStyleColor();  // pop text color here so it only applies to the label
      return m_ui.enumCombobox(GUI_VISUALIZE, "visualize", &m_frameConfig.frameConstants.visualize);
    });
    ImGui::BeginDisabled(m_tweak.renderer != RENDERER_RASTER_CLUSTERS);
    PE::Checkbox("Render cluster bboxes", &m_frameConfig.drawClusterBoxes, "Displays the static original bbox, raster mode only");
    ImGui::EndDisabled();
    PE::end();
  }

  if(ImGui::CollapsingHeader("Clusters & CLAS", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Clusters");
    PE::entry("Cluster/meshlet size",
              [&]() { return m_ui.enumCombobox(GUI_MESHLET, "##HiddenID", &m_tweak.clusterConfig); });
    PE::Checkbox("Use NV cluster library", &m_sceneConfig.clusterNvLibrary,
                 "uses the nv_cluster_builder library, otherwise meshoptimizer");
    PE::InputFloat("NV graph weight", &m_sceneConfig.clusterNvGraphWeight, 0.01f, 0.01f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue,
                   "non-zero weight makes use of triangle connectivity, otherwise disabled");
    PE::InputFloat("NV cost underfill", &m_sceneConfig.clusterNvUnderfill, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "cost to underfilling a cluster");
    PE::InputFloat("NV cost overlap", &m_sceneConfig.clusterNvOverlap, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "cost to bounding box overlap between clusters");
    PE::Checkbox("Optimize for triangle strips", &m_sceneConfig.clusterStripify,
                 "Re-order triangles within cluster optimizing for triangle strips");
    PE::Checkbox("Cluster-dedicated vertices", &m_sceneConfig.clusterDedicatedVertices,
                 "stores vertices per cluster, increases memory / animation work");
    PE::end();

    ImGui::TextDisabled("Ray tracer CLAS specifics");

    PE::begin("##ClustersRTSpecifics");

    ImGui::BeginDisabled(m_tweak.renderer != RENDERER_RAYTRACE_CLUSTERS);

    PE::Checkbox("Use templates", &m_tweak.useTemplates);

    ImGui::BeginDisabled(!m_tweak.useTemplates);
    PE::Checkbox("Use implicit template build", &m_tweak.useImplicitTemplates);
    PE::entry("Template build mode",
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##HiddenID", &m_tweak.templateBuildMode); });
    PE::entry("Template instantiate mode",
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##HiddenID", &m_tweak.templateInstantiateMode); });
    PE::SliderFloat("Template bbox bloat percentage", &m_tweak.templateBboxBloat, -0.001f, 1.0f, "%.3f", 0,
                    "Negative values disable passing template bbox");
    ImGui::EndDisabled();

    ImGui::BeginDisabled(m_tweak.useTemplates);
    PE::entry("CLAS build mode",
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##HiddenID", &m_tweak.clusterBuildMode); });
    ImGui::EndDisabled();

    PE::InputIntClamped("Position truncation bits", (int*)&m_tweak.clusterPositionTruncationBits, 0, 22, 1, 1,
                        ImGuiInputTextFlags_EnterReturnsTrue);

    ImGui::EndDisabled();
    PE::end();
  }

  if(ImGui::CollapsingHeader("Animation", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##Animation");
    PE::Checkbox("Enable animation", &m_rendererConfig.doAnimation);

    ImGui::BeginDisabled(!m_rendererConfig.doAnimation);
    PE::SliderFloat("Override time value", &m_tweak.overrideTime, 0, 10.0f, "%0.3f", 0, "Set to 0 disables override");

    bool ripple = m_frameConfig.frameConstants.animationRippleEnabled != 0;
    PE::Checkbox("Enable ripple deformer", &ripple);
    m_frameConfig.frameConstants.animationRippleEnabled = ripple ? 1 : 0;

    PE::SliderFloat("Ripple frequency", &m_frameConfig.frameConstants.animationRippleFrequency, 0.001f, 200.f);
    PE::SliderFloat("Ripple amplitude", &m_frameConfig.frameConstants.animationRippleAmplitude, 0.f, 0.05f, "%.3f");
    PE::SliderFloat("Ripple speed", &m_frameConfig.frameConstants.animationRippleSpeed, 0.f, 10.f);

    bool twist = m_frameConfig.frameConstants.animationTwistEnabled != 0;
    PE::Checkbox("Enable twist deformer", &twist);
    m_frameConfig.frameConstants.animationTwistEnabled = twist ? 1 : 0;

    PE::SliderFloat("Twist max angle", &m_frameConfig.frameConstants.animationTwistMaxAngle, 0.f, 360.f * 4.f);
    PE::SliderFloat("Twist speed", &m_frameConfig.frameConstants.animationTwistSpeed, 0.f, 2.f);

    ImGui::EndDisabled();
    PE::end();

    ImGui::TextDisabled("Ray tracers specifics");

    PE::begin("##AnimationRTSpecifics");

    ImGui::BeginDisabled(!m_rendererConfig.doAnimation
                         || (m_tweak.renderer != RENDERER_RAYTRACE_TRIANGLES && m_tweak.renderer != RENDERER_RAYTRACE_CLUSTERS));

    TLASUpdateMode updateMode = m_frameConfig.forceTlasFullRebuild ? TLAS_UPDATE_REBUILD : TLAS_UPDATE_REFIT;
    PE::entry("TLAS update", [&]() { return m_ui.enumCombobox(GUI_TLAS_UPDATEMODE, "##HiddenID", &updateMode); });
    m_frameConfig.forceTlasFullRebuild = (updateMode == TLAS_UPDATE_REBUILD);

    ImGui::EndDisabled();

    ImGui::BeginDisabled(!m_rendererConfig.doAnimation || m_tweak.renderer != RENDERER_RAYTRACE_TRIANGLES);
    PE::SliderFloat("BLAS rebuild ratio", &m_frameConfig.blasRebuildFraction, 0.f, 1.f);
    PE::Text("BLAS rebuilds per frame", "%d", int32_t(m_frameConfig.blasRebuildFraction * m_tweak.gridCopies));
    m_frameConfig.blasRebuildFraction = int32_t(m_frameConfig.blasRebuildFraction * m_tweak.gridCopies) / float(m_tweak.gridCopies);
    ImGui::EndDisabled();
    PE::entry("BLAS buildmode", [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##hiddenID", &m_tweak.blasBuildMode); });
    PE::end();
  }

  ImGui::End();

  ImGui::Begin("Statistics");

  if(ImGui::CollapsingHeader("Scene", nullptr, ImGuiTreeNodeFlags_DefaultOpen) && m_renderer)
  {
    if(ImGui::BeginTable("Scene stats", 3, ImGuiTableFlags_None))
    {
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Scene", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Triangles");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numTriangles * m_tweak.gridCopies).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numTriangles).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Clusters");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numClusters * m_tweak.gridCopies).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMetric(m_scene->m_numClusters).c_str());
      ImGui::EndTable();
    }
  }
  if(ImGui::CollapsingHeader("Memory", nullptr, ImGuiTreeNodeFlags_DefaultOpen) && m_renderer)
  {

    Renderer::ResourceUsageInfo resourceInfo = m_renderer->getResourceUsage();

    if(ImGui::BeginTable("Memory stats", 3, ImGuiTableFlags_RowBg))
    {
      ImGui::TableSetupColumn("Memory", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Actual", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Reserved", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("TLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.rtTlasMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(" - ");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("BLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(readback.blasesSize).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.rtBlasMemBytes).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("CLAS");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(readback.clustersSize).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.rtClasMemBytes).c_str());
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Other RT");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.rtOtherMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(" - ");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Scene");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.sceneMemBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text(" - ");
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("Total");
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.rtTlasMemBytes + readback.blasesSize + readback.clustersSize
                                         + resourceInfo.rtOtherMemBytes + resourceInfo.sceneMemBytes)
                            .c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatMemorySize(resourceInfo.getTotalSum()).c_str());
      ImGui::EndTable();
    }
  }
  if(ImGui::CollapsingHeader("Model Clusters "))
  {
    ImGui::Text("Cluster triangle count: %d", m_scene->m_config.clusterTriangles);
    ImGui::Text("Cluster vertex count: %d", m_scene->m_config.clusterVertices);
    ImGui::Text("Cluster count: %d", m_scene->m_numClusters);
    ImGui::Text("Clusters with max (%d) triangles: %d (%.1f%%)", m_scene->m_config.clusterTriangles,
                m_scene->m_clusterTriangleHistogram.back(),
                float(m_scene->m_clusterTriangleHistogram.back()) * 100.f / float(m_scene->m_numClusters));

    uiPlot(std::string("Cluster Triangle Histogram"), std::string("Cluster count with %d triangles: %d"),
           m_scene->m_clusterTriangleHistogram, m_scene->m_clusterTriangleHistogramMax);
    uiPlot(std::string("Cluster Vertex Histogram"), std::string("Cluster count with %d vertices: %d"),
           m_scene->m_clusterVertexHistogram, m_scene->m_clusterVertexHistogramMax);
  }
  ImGui::End();

  ImGui::Begin("Misc Settings");

  if(ImGui::CollapsingHeader("Camera", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGuiH::CameraWidget();
  }

  if(ImGui::CollapsingHeader("Lighting and Shading", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    namespace PE = ImGuiH::PropertyEditor;

    PE::begin();
    PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                    "Mix between flashlight and sun light");
    PE::end();

    ImGui::TextDisabled("Sun & Sky");
    {
      PE::begin();
      nvvkhl::skyParametersUI(m_frameConfig.frameConstants.skyParams);
      PE::end();
    }
  }

  ImGui::End();

#ifdef _DEBUG
  ImGui::Begin("Debug");

  if(ImGui::CollapsingHeader("Misc settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##HiddenID");
    PE::Checkbox("draw objects", &m_frameConfig.drawObjects);
    PE::InputInt("Colorize xor", (int*)&m_frameConfig.frameConstants.colorXor);
    PE::Checkbox("Auto reset timer", &m_tweak.autoResetTimers);
    PE::end();
  }

  if(ImGui::CollapsingHeader("Shader readback Values", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::InputInt("dbgInt", (int*)&m_frameConfig.frameConstants.dbgUint);
    ImGui::InputFloat("dbgFloat", &m_frameConfig.frameConstants.dbgFloat, 0.1f, 1.0f, "%.3f");

    ImGui::Text(" debugI :  %10d", readback.debugI);
    ImGui::Text(" debugUI:  %10u", readback.debugUI);
    static bool debugFloat = false;
    ImGui::Checkbox(" as float", &debugFloat);
    if(debugFloat)
    {
      for(uint32_t i = 0; i < 32; i++)
      {
        ImGui::Text("%2d: %f %f", i, *(float*)&readback.debugA[i], *(float*)&readback.debugB[i]);
      }
    }
    else
    {
      for(uint32_t i = 0; i < 32; i++)
      {
        ImGui::Text("%2d: %10u %10u %10u", i, readback.debugA[i], readback.debugB[i], readback.debugC[i]);
      }
    }
  }
  ImGui::End();
#endif
}

void AnimatedClusters::viewportUI(ImVec2 corner)
{
  if(ImGui::IsItemHovered())
  {
    const auto mouseAbsPos = ImGui::GetMousePos();

    glm::uvec2 mousePos = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

    m_frameConfig.frameConstants.mousePosition = mousePos * glm::uvec2(m_tweak.supersample, m_tweak.supersample);
    m_mouseButtonHandler.update(mousePos);
    MouseButtonHandler::ButtonState leftButtonState = m_mouseButtonHandler.getButtonState(ImGuiMouseButton_Left);
    m_requestCameraRecenter = (leftButtonState == MouseButtonHandler::eDoubleClick) || ImGui::IsKeyPressed(ImGuiKey_Space);
  }
}
}  // namespace animatedclusters
