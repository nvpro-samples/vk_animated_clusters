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

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <implot/implot.h>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/window.hpp>
#include <nvgui/file_dialog.hpp>

#include "animatedclusters.hpp"

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

template <typename T>
void uiPlot(std::string plotName, std::string tooltipFormat, const std::vector<T>& data, const T& maxValue)
{
  ImVec2 plotSize = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2);

  // Ensure minimum height to avoid overly squished graphics
  plotSize.y = std::max(plotSize.y, ImGui::GetTextLineHeight() * 20);

  const ImPlotFlags     plotFlags = ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText | ImPlotFlags_Crosshairs;
  const ImPlotAxisFlags axesFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoLabel;
  const ImColor         plotColor = ImColor(0.07f, 0.9f, 0.06f, 1.0f);

  if(ImPlot::BeginPlot(plotName.c_str(), plotSize, plotFlags))
  {
    ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_NoButtons);
    ImPlot::SetupAxes(nullptr, "Count", axesFlags, axesFlags);
    ImPlot::SetupAxesLimits(0, static_cast<double>(data.size()), 0, static_cast<double>(maxValue), ImPlotCond_Always);

    ImPlot::PushStyleVar(ImPlotStyleVar_FillAlpha, 0.25f);
    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::SetNextFillStyle(plotColor);
    ImPlot::PlotShaded("", data.data(), (int)data.size(), -INFINITY, 1.0, 0.0, 0, 0);
    ImPlot::PopStyleVar();

    if(ImPlot::IsPlotHovered())
    {
      ImPlotPoint mouse       = ImPlot::GetPlotMousePos();
      int         mouseOffset = (int(mouse.x)) % (int)data.size();
      ImGui::BeginTooltip();
      ImGui::Text(tooltipFormat.c_str(), mouseOffset, data[mouseOffset]);
      ImGui::EndTooltip();
    }

    ImPlot::EndPlot();
  }
}

void AnimatedClusters::viewportUI(ImVec2 corner)
{
  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

  m_frameConfig.frameConstants.mousePosition = mousePos * glm::uvec2(m_tweak.supersample, m_tweak.supersample);
}

void AnimatedClusters::onUIRender()
{
  ImGuiWindow* viewport = ImGui::FindWindowByName("Viewport");

  if(viewport)
  {
    if(nvgui::isWindowHovered(viewport))
    {
      if(ImGui::IsKeyDown(ImGuiKey_R))
      {
        m_reloadShaders = true;
      }
      if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
      {
        m_requestCameraRecenter = true;
      }
    }
  }

  bool earlyOut = !m_scene;

  if(earlyOut)
  {
    return;
  }

  shaderio::Readback readback;
  m_resources.getReadbackData(readback);

  // camera control, recenter
  if(m_requestCameraRecenter && isPickingValid(readback))
  {

    glm::uvec2 mousePos = {m_frameConfig.frameConstants.mousePosition.x / m_tweak.supersample,
                           m_frameConfig.frameConstants.mousePosition.y / m_tweak.supersample};

    const glm::mat4 view = m_info.cameraManipulator->getViewMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4       win_norm = {0, 0, m_frameConfig.frameConstants.viewport.x / m_tweak.supersample,
                                  m_frameConfig.frameConstants.viewport.y / m_tweak.supersample};
      const glm::vec3 hitPos   = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      // Set the interest position
      glm::vec3 eye, center, up;
      m_info.cameraManipulator->getLookat(eye, center, up);
      m_info.cameraManipulator->setLookat(eye, hitPos, up, false);
    }

    m_requestCameraRecenter = false;
  }

  // for emphasized parameter we want to recommend to the user
  const ImVec4 recommendedColor = ImVec4(0.0, 1.0, 0.0, 1.0);
  // for warnings
  const ImVec4 warnColor = ImVec4(1.0f, 0.7f, 0.3f, 1.0f);

  namespace PE = nvgui::PropertyEditor;

  ImGui::Begin("Settings");
  ImGui::PushItemWidth(170 * ImGui::GetWindowDpiScale());

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

    bool isRayTracing = m_tweak.renderer == RENDERER_RAYTRACE_CLUSTERS || m_tweak.renderer == RENDERER_RAYTRACE_TRIANGLES;
    if(isRayTracing && m_tweak.supersample > 1)
    {
      ImGui::PushStyleColor(ImGuiCol_Text, warnColor);
    }
    PE::Text("Render Resolution",
             fmt::format("{} x {}", m_resources.m_frameBuffer.renderSize.width, m_resources.m_frameBuffer.renderSize.height));
    if(isRayTracing && m_tweak.supersample > 1)
    {
      ImGui::PopStyleColor();
    }
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
    ImGui::BeginDisabled(!m_sceneConfig.clusterNvLibrary);
    PE::InputFloat("NV cost underfill tris", &m_sceneConfig.clusterNvConfig.costUnderfill, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "cost to underfilling triangles per cluster");
    PE::InputFloat("NV cost underfill verts", &m_sceneConfig.clusterNvConfig.costUnderfillVertices, 0.01f, 0.01f,
                   "%.3f", ImGuiInputTextFlags_EnterReturnsTrue, "cost to underfilling vertices per cluster");
    PE::InputFloat("NV cost overlap", &m_sceneConfig.clusterNvConfig.costOverlap, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "cost to bounding box overlap between clusters");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(m_sceneConfig.clusterNvLibrary);
    PE::InputFloat("Meshopt fill weight", &m_sceneConfig.clusterMeshoptSpatialFill, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "Bias full vs SaH optimal clusters");
    ImGui::EndDisabled();
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
    if(PE::Checkbox("Enable animation", &m_rendererConfig.doAnimation))
    {
      m_tweak.blasBuildMode = m_rendererConfig.doAnimation ? BuildMode::BUILD_FAST_BUILD : BuildMode::BUILD_FAST_TRACE;
      m_tweak.templateInstantiateMode = m_rendererConfig.doAnimation ? BuildMode::BUILD_FAST_BUILD : BuildMode::BUILD_FAST_TRACE;
      m_tweak.clusterBuildMode = m_rendererConfig.doAnimation ? BuildMode::BUILD_FAST_BUILD : BuildMode::BUILD_FAST_TRACE;
      m_tweak.useTemplates = m_rendererConfig.doAnimation;
    }

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
      ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);
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
    ImGui::Text("Cluster max triangle count: %d", m_scene->m_maxClusterTriangles);
    ImGui::Text("Cluster max vertex count: %d", m_scene->m_maxClusterVertices);
    ImGui::Text("Cluster count: %d", m_scene->m_numClusters);
    ImGui::Text("Clusters with config (%d) triangles: %d (%.1f%%)", m_scene->m_config.clusterTriangles,
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
    nvgui::CameraWidget(m_info.cameraManipulator);
  }

  if(ImGui::CollapsingHeader("Lighting and Shading", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    namespace PE = nvgui::PropertyEditor;

    PE::begin();
    PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                    "Mix between flashlight and sun light");
    PE::end();

    ImGui::TextDisabled("Sun & Sky");
    nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams);
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

  handleChanges();

  // Rendered image displayed fully in 'Viewport' window
  ImGui::Begin("Viewport");
  ImVec2 corner = ImGui::GetCursorScreenPos();  // Corner of the viewport
  ImGui::Image((ImTextureID)m_imguiTexture, ImGui::GetContentRegionAvail());
  viewportUI(corner);
  ImGui::End();
}

void AnimatedClusters::onUIMenu()
{
  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem("Open", "Ctrl+O"))
    {
      std::filesystem::path filename =
          nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF", "glTF(.gltf, .glb)|*.gltf;*.glb");
      if(!filename.empty())
      {
        onFileDrop(filename);
      }
    }
    ImGui::EndMenu();
  }
}
}  // namespace animatedclusters
