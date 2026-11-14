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

#include "shaderio.h"
#include "nvshaders/sky_functions.h.slang"

//////////////////////////////////////////////////////////////

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

//////////////////////////////////////////////////////////////

layout(location = RAYTRACING_PAYLOAD_INDEX) rayPayloadInEXT RayPayload rayHit;

//////////////////////////////////////////////////////////////

void main()
{
#if RAYTRACING_PAYLOAD_INDEX == 0
  vec3 skyColor = evalSimpleSky(view.skyParams, gl_WorldRayDirectionEXT);
#else
  vec3 skyColor = vec3(1);
#endif

  rayHit.color.rgb = skyColor;
  rayHit.color.w   = 0.f;
}
