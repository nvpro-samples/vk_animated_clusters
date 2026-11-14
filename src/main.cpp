/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define VMA_IMPLEMENTATION

#include <nvvk/validation_settings.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvapp/elem_sequencer.hpp>

#include "animatedclusters.hpp"

using namespace animatedclusters;

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name    = TARGET_NAME;
  appInfo.useMenu = true;

  VkPhysicalDeviceMeshShaderFeaturesEXT meshEXT = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rayPosKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clustersNV = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};
  VkPhysicalDeviceShaderClockFeaturesKHR clockKHR = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
  };
  vkSetup.deviceExtensions.push_back({VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshEXT});
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accKHR});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rayKHR});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rayPosKHR});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayQueryKHR});
  // set to version 2 compatibility instead of VK_NV_CLUSTER_ACCELERATION_STRUCTURE_SPEC_VERSION to cover more drivers
  vkSetup.deviceExtensions.push_back({VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, &clustersNV, false, 2});
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockKHR});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures});
  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &shadingRateFeatures});


  nvutils::ProfilerManager                    profilerManager;
  std::shared_ptr<nvutils::CameraManipulator> cameraManipulator = std::make_shared<nvutils::CameraManipulator>();
  nvvk::ValidationSettings::LayerPresets      validationPreset  = nvvk::ValidationSettings::LayerPresets::eStandard;

  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              parameterParser;
  nvutils::ParameterSequencer::InitInfo sequencerInfo{// sequencer always requires a parser and registry
                                                      .parameterParser   = &parameterParser,
                                                      .parameterRegistry = &parameterRegistry,
                                                      // sequencer uses the profiler for benchmarking
                                                      .profilerManager = &profilerManager};

  parameterRegistry.add({"validation"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"validationpreset"}, (int*)&validationPreset);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"headless", "run without a window"}, &appInfo.headless, true);
  parameterRegistry.add({"headlessframes", "frames to render in headless mode. default 1"}, &appInfo.headlessFrameCount);
  parameterRegistry.add({"headlessfps", "frame rate limit in headless mode, idles the GPU between frames. 0 unlimited (default)"},
                        &appInfo.headlessFps);
  parameterRegistry.addVector({"windowsize", "window size, or the viewport when headless: width height. default 0 0 picks one"},
                              &appInfo.windowSize);

  AnimatedClusters::Info sampleInfo;
  sampleInfo.cameraManipulator                    = cameraManipulator;
  sampleInfo.profilerManager                      = &profilerManager;
  sampleInfo.parameterRegistry                    = &parameterRegistry;
  std::shared_ptr<AnimatedClusters> sampleElement = std::make_shared<AnimatedClusters>(sampleInfo);

  // add a few more parameters to registry and parser to handle sequencer settings
  sequencerInfo.registerScriptParameters(parameterRegistry, parameterParser);

  // takes a screenshot at the end of each sequence
  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& state) { sampleElement->parameterSequenceCallback(state); });

  parameterParser.add(parameterRegistry);
  parameterParser.parse(argc, argv);

  // this element requires sequencerInfo that is potentially updated by parameterParser
  auto elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);

  nvvk::ValidationSettings validationSettings;
  if(vkSetup.enableValidationLayers)
  {
    validationSettings.setPreset(validationPreset);
    validationSettings.duplicate_message_limit = 3;
    // some false positives
    validationSettings.message_id_filter = {"VUID-RuntimeSpirv-storageInputOutput16-06334", "VUID-VkShaderModuleCreateInfo-pCode-08740",
                                            "VUID-vkCmdBuildClusterAccelerationStructureIndirectNV-pCommandInfos-12307",
                                            "VUID-vkCmdBuildClusterAccelerationStructureIndirectNV-dstImplicitData-12303",
                                            // layer reads the opInput union as triangle input for move objects
                                            "VUID-VkClusterAccelerationStructureTriangleClusterInputNV-sType-sType",
                                            "VUID-VkClusterAccelerationStructureTriangleClusterInputNV-vertexFormat-10439",
                                            "VUID-VkClusterAccelerationStructureTriangleClusterInputNV-maxClusterVertexCount-10441"};

    vkSetup.instanceCreateInfoExt = validationSettings.buildPNextChain();
  }

  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  sampleElement->setSupportsClusters(vkContext.hasExtensionEnabled(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME));

  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
#ifndef NDEBUG
    // left side panel container
    ImGuiID debugID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.15F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Debug", debugID);
#endif
    // right side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGui::DockBuilderDockWindow("Misc Settings", settingID);

    // bottom panel container
    ImGuiID loggerID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", loggerID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(loggerID, ImGuiDir_Right, 0.4F, nullptr, &loggerID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID statisticsID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Right, 0.5F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("Statistics", statisticsID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  auto                  logger      = std::make_shared<nvapp::ElementLogger>();
  nvapp::ElementLogger* loggerDeref = logger.get();
  nvutils::Logger::getInstance().setLogCallback([&](nvutils::Logger::LogLevel logLevel, const std::string& text) {
    loggerDeref->addLog(logLevel, "%s", text.c_str());
  });

  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  app.addElement(logger);
  app.addElement(sampleElement);
  app.addElement(std::make_shared<nvapp::ElementCamera>(cameraManipulator));
  app.addElement(std::make_shared<nvapp::ElementProfiler>(&profilerManager));
  app.addElement(elemSequencer);
  app.run();

  nvutils::Logger::getInstance().setLogCallback(nullptr);

  // Cleanup in reverse order
  app.deinit();
  vkContext.deinit();

  return 0;
}
