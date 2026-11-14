/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"

layout(push_constant) uniform pushData
{
  uint instanceID;
} push;

layout(scalar, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
{
  FrameConstants view;
};

layout(scalar, binding = BINDINGS_READBACK_SSBO, set = 0) buffer readbackBuffer
{
  Readback readback;
};

layout(scalar, binding = BINDINGS_RENDERINSTANCES_SSBO, set = 0) buffer renderInstancesBuffer
{
  RenderInstance instances[];
};

////////////////////////////////////////////

layout(location=0) out Interpolants {
  vec3 wPos;
  vec3 wNormal;
  flat uint clusterID;
} OUT;

////////////////////////////////////////////

void main()
{
  vec3s_in oPositions = vec3s_in(instances[push.instanceID].positions);
  vec3s_in oNormals   = vec3s_in(instances[push.instanceID].normals);
  
  mat4 worldMatrix = instances[push.instanceID].worldMatrix;
  
  vec3 oPos = oPositions.d[gl_VertexIndex];
  vec4 wPos = worldMatrix * vec4(oPos,1.0f);
  
  mat3 worldMatrixIT = transpose(inverse(mat3(worldMatrix)));
  
  gl_Position = view.viewProjMatrix * wPos;
  OUT.wPos    = wPos.xyz;
  
  vec3 oNormal = oNormals.d[gl_VertexIndex];
  OUT.wNormal  = normalize(worldMatrixIT * oNormal);
  
  OUT.clusterID = 0;
}
