/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_ray_tracing : require

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference   :enable
#extension GL_EXT_scalar_block_layout:enable
#include "shaderio.h"

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(set = 0, binding = BINDINGS_TLAS)          uniform accelerationStructureEXT asScene;
layout(set = 0, binding = BINDINGS_RENDER_TARGET, rgba8)  uniform image2D imgColor;

//////////////////////////////////////////////////////////////

layout(location = 0) rayPayloadEXT RayPayload rayHit;

//////////////////////////////////////////////////////////////

void main()
{
  // for writing debugging values to stats.debug etc.
  bool  center = gl_LaunchIDEXT.xy == (gl_LaunchSizeEXT.xy/2);
  
  ivec2 screen = ivec2(gl_LaunchIDEXT.xy);
  vec2 uv = (vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy);
  //uv.y = 1.f-uv.y;
  //uv.x = 1.f-uv.x;
  vec2 d = uv * 2.0 - 1.0;
  vec4 origin    = view.viewMatrixI * vec4(0, 0, 0, 1);
  vec3 target    = normalize((view.projMatrixI * vec4(d.x, d.y, 1, 1)).xyz);
  vec4 direction = view.viewMatrixI * vec4(target, 0);

  float tMin     = view.nearPlane;
  float tMax     = view.farPlane;

  traceRayEXT(asScene, gl_RayFlagsCullBackFacingTrianglesEXT,
    0xff,
    0, 0, // hit offset, hit stride
    0,    // miss offset
    origin.xyz, tMin, direction.xyz, tMax,
    0     // rayPayloadNV location qualifier
  );

  {
    imageStore(imgColor, screen, vec4(rayHit.color.xyz,1));
  }
}