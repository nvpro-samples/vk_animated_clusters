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

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_fragment_shader_barycentric : enable

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

///////////////////////////////////////////////////

#include "render_shading.glsl"

///////////////////////////////////////////////////

layout(location = 0) in Interpolants
{
  vec3      wPos;
  vec3      wNormal;
  flat uint clusterID;
}
IN;

///////////////////////////////////////////////////

layout(location = 0, index = 0) out vec4 out_Color;

///////////////////////////////////////////////////

void main()
{
  vec3 wNormal;

  if(view.facetShading != 0)
  {
    wNormal = -cross(dFdx(IN.wPos), dFdy(IN.wPos));
  }
  else
  {
    wNormal = IN.wNormal;
  }

  uint visClusterID = IN.clusterID;
#if LINKED_MESH_SHADER
  if (view.visualize == VISUALIZE_TRIANGLES)
  {
    visClusterID ^= gl_PrimitiveID + 1;
  }
#endif

  const float overHeadLight = 1.0f;
  const float ambientLight  = 1.f;

  out_Color = shading(push.instanceID, IN.wPos, wNormal, visClusterID, overHeadLight, ambientLight);

#if DEBUG_VISUALIZATION
  if(view.doWireframe != 0 || (view.visFilterInstanceID == push.instanceID && view.visFilterClusterID == IN.clusterID))
  {
    out_Color.xyz = addWireframe(out_Color.xyz, gl_BaryCoordEXT, gl_FrontFacing, fwidthFine(gl_BaryCoordEXT), view.wireColor);
  }
#endif

  uvec2 pixelCoord = uvec2(gl_FragCoord.xy);
  if(pixelCoord == view.mousePosition)
  {
    uint32_t packedClusterTriangleId = (IN.clusterID << 8) | (gl_PrimitiveID & 0xFF);
    atomicMax(readback.clusterTriangleId, packPickingValue(packedClusterTriangleId, gl_FragCoord.z));
    atomicMax(readback.instanceId, packPickingValue(push.instanceID, gl_FragCoord.z));
  }
}