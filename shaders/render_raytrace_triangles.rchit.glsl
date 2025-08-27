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

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

#include "shaderio.h"

/////////////////////////////////

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

layout(set = 0, binding = BINDINGS_TLAS) uniform accelerationStructureEXT asScene;

/////////////////////////////////

hitAttributeEXT vec2 barycentrics;

/////////////////////////////////

layout(location = 0) rayPayloadInEXT RayPayload rayHit;
layout(location = 1) rayPayloadEXT RayPayload rayHitAO;

/////////////////////////////////

#define SUPPORTS_RT 1

#include "render_shading.glsl"

/////////////////////////////////

void main()
{
  // Fetch triangle
  uvec3s_in indexBuffer     = uvec3s_in(instances[gl_InstanceID].triangles);
  uvec3     triangleIndices = indexBuffer.d[gl_PrimitiveID];

  // Fetch vertex positions
  vec3     vertices[3];
  vec3s_in vertexBuffer = vec3s_in(instances[gl_InstanceID].positions);

  for(uint32_t i = 0; i < 3; i++)
  {
    vertices[i] = vertexBuffer.d[triangleIndices[i]];
  }

  vec3 baryWeight = vec3((1.f - barycentrics[0] - barycentrics[1]), barycentrics[0], barycentrics[1]);

  vec3 oPos = baryWeight.x * vertices[0] + baryWeight.y * vertices[1] + baryWeight.z * vertices[2];
  vec3 wPos = vec3(gl_ObjectToWorldEXT * vec4(oPos, 1.0));

  vec3 oNrm;
  if(view.facetShading != 0)
  {
    // Otherwise compute geometric normal
    vec3 e0 = vertices[1] - vertices[0];
    vec3 e1 = vertices[2] - vertices[0];
    oNrm    = normalize(cross(e0, e1));
  }
  else
  {
    vec3     normals[3];
    vec3s_in normalsBuffer = vec3s_in(instances[gl_InstanceID].normals);

    for(uint32_t i = 0; i < 3; i++)
    {
      normals[i] = normalize(normalsBuffer.d[triangleIndices[i]]);
    }
    oNrm = baryWeight.x * normals[0] + baryWeight.y * normals[1] + baryWeight.z * normals[2];
  }

  vec3 wNrm = normalize(vec3(oNrm * gl_WorldToObjectEXT));


  // triangles don't have clusterID
  uint32_t visClusterID = 0;
  if (view.visualize == VISUALIZE_TRIANGLES) {
    visClusterID = 1 + gl_PrimitiveID;
  }

  vec3 directionToLight = view.skyParams.sunDirection;
  float ambientOcclusion = ambientOcclusion(wPos, wNrm, view.ambientOcclusionRays, view.ambientOcclusionRadius * view.sceneSize);

  float sunContribution = 1.0;
  if(view.doShadow == 1)
    sunContribution = traceShadowRay(wPos, wNrm, directionToLight);

  rayHit.color = shading(gl_InstanceID, wPos, wNrm, visClusterID, sunContribution, ambientOcclusion);

  if(gl_LaunchIDEXT.xy == view.mousePosition)
  {
    vec4  projected            = (view.viewProjMatrix * vec4(wPos, 1.f));
    float depth                = projected.z / projected.w;
    readback.clusterTriangleId = packPickingValue(gl_PrimitiveID, depth);
    readback.instanceId        = packPickingValue(gl_InstanceID, depth);
  }
}