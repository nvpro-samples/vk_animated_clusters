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
#extension GL_EXT_control_flow_attributes : require

// at the time of writing, no GLSL extension was available, we leverage
// GL_EXT_spirv_intrinsics to hook up the new builtin.
#extension GL_EXT_spirv_intrinsics : require

// Note that `VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV::allowClusterAccelerationStructures` must
// be set to `VK_TRUE` to make this valid.
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

// While not required in this sample, as we use dedicated hit-shader for clusters,
// `int gl_ClusterIDNoneNV = -1;` can be used to dynamically detect regular hits.

#include "shaderio.h"

/////////////////////////////////

layout(std140, binding = BINDINGS_FRAME_UBO, set = 0) uniform frameConstantsBuffer
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

#ifndef CLUSTER_DEDICATED_VERTICES
#define CLUSTER_DEDICATED_VERTICES 0
#endif

/////////////////////////////////

void main()
{
  // get cluster ID (see top of file how we hooked up this value to spir-v)
  uint clusterID = gl_ClusterIDNV_;

  uint visClusterID = clusterID;
  if (view.visualize == VISUALIZE_TRIANGLES) {
    visClusterID ^= 1 + gl_PrimitiveID;
  }

  RenderInstance instance = instances[gl_InstanceID];

  // Fetch cluster header
  Clusters_in clusterBuffer = Clusters_in(instance.clusters);
  Cluster     cluster       = clusterBuffer.d[clusterID];

  // Fetch triangle
  // There is three different possibilities.
#if CLUSTER_DEDICATED_VERTICES
  // The data has been baked to have vertices per-cluster.
  // This way we get away with the 8-bit triangle indices that are local to the cluster.

  // the local triangle indices used within this cluster
  uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);

  uvec3 triangleIndices = uvec3(localTriangles.d[cluster.firstLocalTriangle + gl_PrimitiveID * 3 + 0],
                                localTriangles.d[cluster.firstLocalTriangle + gl_PrimitiveID * 3 + 1],
                                localTriangles.d[cluster.firstLocalTriangle + gl_PrimitiveID * 3 + 2]);

  // convert to global indices for attribute lookup
  triangleIndices += cluster.firstLocalVertex;

#elif (!CLUSTER_DEDICATED_VERTICES) && 0
  // Disable this for codepath for now, given we kept the original indexbuffer for computing the normals anyway,
  // and disabling avoids the indirection. When the original triangle indexbuffer isn't needed
  // then using this would be less memory.
  
  // the local triangle indices used within this cluster
  uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);

  uvec3 triangleIndices = uvec3(localTriangles.d[cluster.firstLocalTriangle + gl_PrimitiveID * 3 + 0],
                                localTriangles.d[cluster.firstLocalTriangle + gl_PrimitiveID * 3 + 1],
                                localTriangles.d[cluster.firstLocalTriangle + gl_PrimitiveID * 3 + 2]);

  // convert to global indices for attribute lookup
  
  // we need another indirection, mapping the local triangle indices, to the global
  // vertex indices within the cluster.
  uints_in  localVertices  = uints_in(instance.clusterLocalVertices);
  
  triangleIndices.x = localVertices.d[cluster.firstLocalVertex + triangleIndices.x];
  triangleIndices.y = localVertices.d[cluster.firstLocalVertex + triangleIndices.y];
  triangleIndices.z = localVertices.d[cluster.firstLocalVertex + triangleIndices.z];

#else
  // The simple way is we just use the traditional triangle index buffer,
  // which operates on global indices already.

  // get the classic triangle index buffer of this instance
  uvec3s_in indexBuffer = uvec3s_in(instance.triangles);
  // fetch triangle with cluster's offset
  // gl_PrimitiveID is the local triangle index within the cluster
  uvec3 triangleIndices = indexBuffer.d[gl_PrimitiveID + cluster.firstTriangle];
#endif

  // Fetch vertex positions
  vec3     vertices[3];
  vec3s_in vertexBuffer = vec3s_in(instance.positions);

  [[unroll]]
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

    [[unroll]]
    for(uint32_t i = 0; i < 3; i++)
    {
      normals[i] = normalize(normalsBuffer.d[triangleIndices[i]]);
    }
    oNrm = baryWeight.x * normals[0] + baryWeight.y * normals[1] + baryWeight.z * normals[2];
  }

  vec3 wNrm = normalize(vec3(oNrm * gl_WorldToObjectEXT));

  vec3 directionToLight = view.skyParams.directionToLight;
  float ambientOcclusion = ambientOcclusion(wPos, wNrm, view.ambientOcclusionRays, view.ambientOcclusionRadius * view.sceneSize);

  float sunContribution = 1.0;
  if(view.doShadow == 1)
    sunContribution = traceShadowRay(wPos, directionToLight);

  rayHit.color = shading(gl_InstanceID, wPos, wNrm, visClusterID, sunContribution, ambientOcclusion);

  if(gl_LaunchIDEXT.xy == view.mousePosition)
  {
    vec4  projected            = (view.viewProjMatrix * vec4(wPos, 1.f));
    float depth                = projected.z / projected.w;
    readback.clusterTriangleId = packPickingValue((clusterID << 8) | gl_PrimitiveID, depth);
    readback.instanceId        = packPickingValue(gl_InstanceID, depth);
  }
}