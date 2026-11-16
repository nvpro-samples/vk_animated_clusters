/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <filesystem>

#include <fmt/chrono.h>
#include <nvutils/file_operations.hpp>

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

    ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
    ImPlot::PlotShaded("", data.data(), (int)data.size(), -INFINITY, 1.0, 0.0,
                       ImPlotSpec(ImPlotProp_FillColor, (ImU32)plotColor, ImPlotProp_FillAlpha, 0.75f));

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

void AnimatedClusters::viewportUI(ImVec2 corner, ImVec2 imageSize)
{
  ImVec2     mouseAbsPos = ImGui::GetMousePos();
  glm::uvec2 mousePos    = {mouseAbsPos.x - corner.x, mouseAbsPos.y - corner.y};

  // the viewport shows the render target scaled to the displayed rect, fixed resolutions are letterboxed
  const Resources::FrameBuffer& fb = m_resources.m_frameBuffer;
  glm::vec2  displaySize = fb.fixedSize ? glm::vec2(std::max(imageSize.x, 1.0f), std::max(imageSize.y, 1.0f)) :
                                          glm::vec2(std::max(fb.windowSize.width, 1u), std::max(fb.windowSize.height, 1u));
  glm::vec2  scale       = glm::vec2(fb.renderSize.width, fb.renderSize.height) / displaySize;
  glm::uvec2 maxPos      = {std::max(fb.renderSize.width, 1u) - 1, std::max(fb.renderSize.height, 1u) - 1};

  m_frameConfig.frameConstants.mousePosition = glm::min(glm::uvec2(glm::vec2(mousePos) * scale), maxPos);
}

void AnimatedClusters::cameraPathUI()
{
  namespace PE = nvgui::PropertyEditor;

  // File: Save / Load the per-scene paths file stored next to the executable
  const std::filesystem::path pathsFile = sceneCameraPathsFile(m_sceneFilePath);
  ImGui::TextUnformatted("File:");
  ImGui::SameLine();
  ImGui::BeginDisabled(pathsFile.empty());
  if(ImGui::SmallButton("Save"))
  {
    if(saveCameraPaths(pathsFile, m_cameraPaths))
      LOGI("Saved %zu camera path(s) to %s\n", m_cameraPaths.size(), nvutils::utf8FromPath(pathsFile).c_str());
    else
      LOGE("Could not write camera paths to %s\n", nvutils::utf8FromPath(pathsFile).c_str());
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Load") && loadCameraPaths(pathsFile, m_cameraPaths))
  {
    // the scene's file replaced the paths, later scene loads use it again
    m_cameraPathsExternal = false;
    m_cameraPathActive = m_cameraPaths.empty() ? -1 : std::min(std::max(m_cameraPathActive, 0), int(m_cameraPaths.size()) - 1);
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::EndDisabled();
  if(pathsFile.empty())
    ImGui::TextDisabled("(load a scene to save/load its paths file)");
  else
    ImGui::TextDisabled("%s", nvutils::utf8FromPath(pathsFile.filename()).c_str());

  ImGui::Spacing();

  // Path: New / Delete / Copy / Paste (Copy/Paste exchange the path string via the clipboard)
  ImGui::TextUnformatted("Path:");
  ImGui::SameLine();
  if(ImGui::SmallButton("New##path"))
  {
    m_cameraPaths.emplace_back();
    m_cameraPathActive   = int(m_cameraPaths.size()) - 1;
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(m_cameraPathActive < 0 || m_cameraPathActive >= int(m_cameraPaths.size()));
  if(ImGui::SmallButton("Delete##path"))
  {
    m_cameraPaths.erase(m_cameraPaths.begin() + m_cameraPathActive);
    m_cameraPathActive   = m_cameraPaths.empty() ? -1 : std::min(m_cameraPathActive, int(m_cameraPaths.size()) - 1);
    m_cameraPathEditKey  = -1;
    m_cameraPathPlayback = CAMERA_PATH_STOP;
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Copy") && m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size()))
    ImGui::SetClipboardText(m_cameraPaths[m_cameraPathActive].getString().c_str());
  ImGui::EndDisabled();
  ImGui::SameLine();
  if(ImGui::SmallButton("Paste"))
  {
    const char* clip = ImGui::GetClipboardText();
    CameraPath  pasted;
    if(clip && pasted.setFromString(clip))
    {
      m_cameraPaths.push_back(std::move(pasted));
      m_cameraPathActive   = int(m_cameraPaths.size()) - 1;
      m_cameraPathEditKey  = -1;
      m_cameraPathPlayback = CAMERA_PATH_STOP;
    }
  }

  // active path selector
  {
    std::string preview = (m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size())) ?
                              fmt::format("Path {} ({} keys)", m_cameraPathActive, m_cameraPaths[m_cameraPathActive].size()) :
                              std::string("<none>");
    ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::BeginCombo("##camerapathsel", preview.c_str()))
    {
      for(int i = 0; i < int(m_cameraPaths.size()); i++)
      {
        if(ImGui::Selectable(fmt::format("Path {} ({} keys)", i, m_cameraPaths[i].size()).c_str(), i == m_cameraPathActive))
        {
          m_cameraPathActive   = i;
          m_cameraPathEditKey  = -1;
          m_cameraPathPlayback = CAMERA_PATH_STOP;
          m_cameraPathU        = 0.0;
        }
      }
      ImGui::EndCombo();
    }
  }

  const bool hasPath = m_cameraPathActive >= 0 && m_cameraPathActive < int(m_cameraPaths.size());
  const nvutils::CameraManipulator::Camera& cam = m_info.cameraManipulator->getCamera();

  // Key: New / Update / Delete (captured from the current camera)
  ImGui::TextUnformatted("Key:");
  ImGui::SameLine();
  ImGui::BeginDisabled(!hasPath);
  if(ImGui::SmallButton("New##key") && hasPath)
  {
    CameraPath& keyPath = m_cameraPaths[m_cameraPathActive];
    int insertAt = (m_cameraPathEditKey >= 0 && m_cameraPathEditKey < int(keyPath.keys.size())) ? m_cameraPathEditKey + 1 :
                                                                                                  int(keyPath.keys.size());
    keyPath.keys.insert(keyPath.keys.begin() + insertAt, CameraPathKey{cam.eye, cam.ctr, cam.up, cam.fov});
    m_cameraPathEditKey = insertAt;
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Update##key") && hasPath && m_cameraPathEditKey >= 0
     && m_cameraPathEditKey < int(m_cameraPaths[m_cameraPathActive].keys.size()))
  {
    m_cameraPaths[m_cameraPathActive].keys[m_cameraPathEditKey] = CameraPathKey{cam.eye, cam.ctr, cam.up, cam.fov};
  }
  ImGui::SameLine();
  if(ImGui::SmallButton("Delete##key") && hasPath && m_cameraPathEditKey >= 0
     && m_cameraPathEditKey < int(m_cameraPaths[m_cameraPathActive].keys.size()))
  {
    CameraPath& keyPath = m_cameraPaths[m_cameraPathActive];
    keyPath.keys.erase(keyPath.keys.begin() + m_cameraPathEditKey);
    m_cameraPathEditKey = std::min(m_cameraPathEditKey, int(keyPath.keys.size()) - 1);
  }
  ImGui::EndDisabled();

  // keyframe list (always shown, even when empty; selecting previews the camera)
  if(ImGui::BeginListBox("##keys", ImVec2(-FLT_MIN, ImGui::GetTextLineHeightWithSpacing() * 4.5f)))
  {
    if(hasPath)
    {
      const CameraPath& selPath = m_cameraPaths[m_cameraPathActive];
      for(int i = 0; i < int(selPath.keys.size()); i++)
      {
        const CameraPathKey& k = selPath.keys[i];
        if(ImGui::Selectable(
               fmt::format("{:2d}: eye {:.1f} {:.1f} {:.1f}  fov {:.0f}", i, k.eye.x, k.eye.y, k.eye.z, k.fov).c_str(),
               i == m_cameraPathEditKey))
        {
          m_cameraPathEditKey                  = i;
          m_cameraPathPlayback                 = CAMERA_PATH_STOP;
          nvutils::CameraManipulator::Camera c = cam;
          c.eye = k.eye, c.ctr = k.ctr, c.up = k.up, c.fov = k.fov;
          m_info.cameraManipulator->setCamera(c, false);  // animate for a nicer preview
        }
      }
    }
    ImGui::EndListBox();
  }

  if(!hasPath)
    return;

  CameraPath& path = m_cameraPaths[m_cameraPathActive];

  PE::begin("camerapath", ImGuiTableFlags_Resizable);
  PE::Checkbox("Smooth", &path.smooth, "Catmull-Rom interpolation, otherwise piecewise linear");
  PE::Checkbox("Loop", &path.loop, "Real-time playback wraps around");
  {
    float dur = float(path.duration);
    if(PE::InputFloat("Duration (s)", &dur, 0.1f, 1.0f, "%.2f", 0, "Seconds for a full real-time playback"))
      path.duration = std::max(double(dur), 0.0);
  }
  PE::end();

  // real-time playback + scrub
  const bool active = m_cameraPathPlayback != CAMERA_PATH_STOP;
  if(ImGui::Button(active ? "Stop" : "Play"))
  {
    if(active)
    {
      m_cameraPathPlayback = CAMERA_PATH_STOP;
    }
    else
    {
      m_cameraPathPlayback = CAMERA_PATH_REALTIME;
      m_cameraPathStarted  = false;
      if(m_cameraPathU >= 1.0)
        m_cameraPathU = 0.0;
    }
  }
  ImGui::SameLine();
  if(ImGui::Button("Restart"))
  {
    m_cameraPathU       = 0.0;
    m_cameraPathStarted = false;
  }
  ImGui::SameLine();
  {
    float u = float(m_cameraPathU);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if(ImGui::SliderFloat("##scrub", &u, 0.0f, 1.0f, "t = %.3f"))
    {
      m_cameraPathU        = u;
      m_cameraPathPlayback = CAMERA_PATH_STOP;
      applyCameraPathSample(m_cameraPathU);
    }
  }

  // deterministic (fixed-step) playback, same behaviour as --runcamerapath
  PE::begin("camerapathfixed", ImGuiTableFlags_Resizable);
  PE::InputInt("Fixed frames", &m_cameraPathFrames, 1, 16, 0,
               "Number of frames for a full fixed-step (deterministic) traversal, as used by --runcamerapath");
  PE::end();
  m_cameraPathFrames = std::max(m_cameraPathFrames, 1);
  if(ImGui::Button("Run fixed"))
  {
    m_cameraPathFrame    = 0;
    m_cameraPathPlayback = CAMERA_PATH_FIXED;
  }
  ImGui::SameLine();
  if(m_cameraPathPlayback == CAMERA_PATH_FIXED)
    ImGui::Text("frame %d / %d", m_cameraPathFrame, m_cameraPathFrames);
  else
    ImGui::TextDisabled("cmd: --runcamerapath %d %d", m_cameraPathActive, m_cameraPathFrames);
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

      bool screenShotJpg = ImGui::IsKeyChordPressed(ImGuiKey_F12);
      bool screenShotPng = ImGui::IsKeyChordPressed(ImGuiMod_Shift | ImGuiKey_F12);
      if(screenShotJpg || screenShotPng)
      {
        auto        now      = std::chrono::system_clock::now();
        std::string filename = fmt::format("screenshot_{:%Y_%m_%d_%H_%M_%S}.{}", now, screenShotJpg ? "jpg" : "png");
        saveScreenshot(SCREENSHOT_VIEWPORT, filename, screenShotJpg ? 90 : 100);
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

    // picking works in render target pixels
    glm::uvec2 mousePos = m_frameConfig.frameConstants.mousePosition;

    const glm::mat4 view = m_info.cameraManipulator->getViewMatrix();
    const glm::mat4 proj = m_frameConfig.frameConstants.projMatrix;

    float d = decodePickingDepth(readback);

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4 win_norm     = {0, 0, m_frameConfig.frameConstants.viewport.x, m_frameConfig.frameConstants.viewport.y};
      const glm::vec3 hitPos = glm::unProjectZO({mousePos.x, mousePos.y, d}, view, proj, win_norm);

      // Set the interest position
      glm::dvec3 eye, center, up;
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
    const bool isScaled = m_resources.m_frameBuffer.renderSize.width != m_resources.m_frameBuffer.windowSize.width
                          || m_resources.m_frameBuffer.renderSize.height != m_resources.m_frameBuffer.windowSize.height;
    if(isRayTracing && isScaled)
    {
      ImGui::PushStyleColor(ImGuiCol_Text, warnColor);
    }
    PE::Text("Render Resolution",
             fmt::format("{} x {}", m_resources.m_frameBuffer.renderSize.width, m_resources.m_frameBuffer.renderSize.height));
    if(isRayTracing && isScaled)
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
        PE::InputFloat("Radius", &m_tweak.hbaoRadius, 0.01f);
        PE::Checkbox("Blur", &m_frameConfig.hbaoSettings.blur);
        PE::Checkbox("Quarter resolution", &m_tweak.hbaoQuarter, "Half width and height, upsampled depth-aware");
        PE::InputFloat("Power", &m_frameConfig.hbaoSettings.powerExponent, 1.0f);
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
              [&]() { return m_ui.enumCombobox(GUI_MESHLET, "##ClusterSize", &m_tweak.clusterConfig); });
    PE::Checkbox("Use spatial (RT)", &m_sceneConfig.clusterSpatial, "uses spatial builder, otherwise flex builder");
    ImGui::BeginDisabled(!m_sceneConfig.clusterSpatial);
    PE::InputFloat("Spatial fill weight", &m_sceneConfig.clusterMeshoptSpatialFill, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "Bias full vs SaH optimal clusters using spatial clustering.");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(m_sceneConfig.clusterSpatial);
    PE::InputFloat("Flex split factor", &m_sceneConfig.clusterMeshoptFlexSplit, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "Split a cluster if its relative spherical dimension exceeds this factor");
    PE::InputFloat("Flex cone weight", &m_sceneConfig.clusterMeshoptFlexCone, 0.01f, 0.01f, "%.3f",
                   ImGuiInputTextFlags_EnterReturnsTrue, "Influence of normal cone weight when clustering");
    ImGui::EndDisabled();
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
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##TemplateBuildMode", &m_tweak.templateBuildMode); });
    PE::entry("Template instantiate mode", [&]() {
      return m_ui.enumCombobox(GUI_BUILDMODE, "##TemplateInstantiateMode", &m_tweak.templateInstantiateMode);
    });
    PE::SliderFloat("Template bbox bloat percentage", &m_tweak.templateBboxBloat, -0.001f, 1.0f, "%.3f", 0,
                    "Negative values disable passing template bbox");
    ImGui::EndDisabled();

    ImGui::BeginDisabled(m_tweak.useTemplates);
    PE::entry("CLAS build mode",
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##ClasBuildMode", &m_tweak.clusterBuildMode); });
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
    PE::entry("TLAS update", [&]() { return m_ui.enumCombobox(GUI_TLAS_UPDATEMODE, "##TlasUpdate", &updateMode); });
    m_frameConfig.forceTlasFullRebuild = (updateMode == TLAS_UPDATE_REBUILD);

    ImGui::EndDisabled();

    ImGui::BeginDisabled(!m_rendererConfig.doAnimation || m_tweak.renderer != RENDERER_RAYTRACE_TRIANGLES);
    PE::SliderFloat("BLAS rebuild ratio", &m_frameConfig.blasRebuildFraction, 0.f, 1.f);
    PE::Text("BLAS rebuilds per frame", "%d", int32_t(m_frameConfig.blasRebuildFraction * m_tweak.gridCopies));
    m_frameConfig.blasRebuildFraction = int32_t(m_frameConfig.blasRebuildFraction * m_tweak.gridCopies) / float(m_tweak.gridCopies);
    ImGui::EndDisabled();
    PE::entry("BLAS buildmode",
              [&]() { return m_ui.enumCombobox(GUI_BUILDMODE, "##BlasBuildMode", &m_tweak.blasBuildMode); });
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
  if(ImGui::CollapsingHeader("Camera Paths", nullptr))
  {
    cameraPathUI();
  }

  if(ImGui::CollapsingHeader("Misc", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    PE::begin("##MiscSettings");
    PE::Checkbox("Auto settings", &m_autoSettings,
                 "On animation changes, sets the build modes and template usage that suit the animation state");
    PE::end();
  }

  if(ImGui::CollapsingHeader("Lighting and Shading", nullptr, ImGuiTreeNodeFlags_DefaultOpen))
  {
    namespace PE = nvgui::PropertyEditor;

    PE::begin("##Lighting");
    PE::SliderFloat("Light Mixer", &m_frameConfig.frameConstants.lightMixer, 0.0f, 1.0f, "%.3f", 0,
                    "Mix between flashlight and sun light");
    PE::end();

    ImGui::TextDisabled("Sun & Sky");
    nvgui::skySimpleParametersUI(m_frameConfig.frameConstants.skyParams, "##Sky");
  }

  ImGui::End();

#ifndef NDEBUG
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
  ImVec2 avail     = ImGui::GetContentRegionAvail();
  ImVec2 imageSize = avail;
  ImVec2 corner    = ImGui::GetCursorScreenPos();  // Corner of the viewport

  const Resources::FrameBuffer& fb = m_resources.m_frameBuffer;
  if(fb.fixedSize && fb.renderSize.width && fb.renderSize.height)
  {
    // fixed resolution: fit its aspect into the viewport, centered
    float scale = std::min(avail.x / float(fb.renderSize.width), avail.y / float(fb.renderSize.height));
    imageSize   = {float(fb.renderSize.width) * scale, float(fb.renderSize.height) * scale};
    corner.x += std::max(0.0f, (avail.x - imageSize.x) * 0.5f);
    corner.y += std::max(0.0f, (avail.y - imageSize.y) * 0.5f);
    ImGui::SetCursorScreenPos(corner);
  }
  ImGui::Image((ImTextureID)m_imguiTexture, imageSize);
  viewportUI(corner, imageSize);
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
