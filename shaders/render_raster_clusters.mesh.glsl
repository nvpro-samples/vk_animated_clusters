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

#extension GL_EXT_mesh_shader : require
#extension GL_EXT_control_flow_attributes: require

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
} OUT[];

////////////////////////////////////////////

#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32
#endif

#ifndef CLUSTER_VERTEX_COUNT
#define CLUSTER_VERTEX_COUNT 32
#endif

#ifndef CLUSTER_TRIANGLE_COUNT
#define CLUSTER_TRIANGLE_COUNT 32
#endif

#ifndef CLUSTER_DEDICATED_VERTICES
#define CLUSTER_DEDICATED_VERTICES 0
#endif

layout(local_size_x=32) in;
layout(max_vertices=CLUSTER_VERTEX_COUNT, max_primitives=CLUSTER_TRIANGLE_COUNT) out;
layout(triangles) out;

const uint MESHLET_VERTEX_ITERATIONS   = ((CLUSTER_VERTEX_COUNT   + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);
const uint MESHLET_TRIANGLE_ITERATIONS = ((CLUSTER_TRIANGLE_COUNT + MESHSHADER_WORKGROUP_SIZE - 1) / MESHSHADER_WORKGROUP_SIZE);

////////////////////////////////////////////

void main()
{
  RenderInstance instance = instances[push.instanceID];
  
  Cluster cluster = Clusters_in(instance.clusters).d[gl_WorkGroupID.x];
  
  uint vertMax = cluster.numVertices-1;
  uint triMax  = cluster.numTriangles-1;

  // We keep things simple and avoid per-triangle culling. It reduces
  // the complexity of the mesh shader and may not always be worth it.

  SetMeshOutputsEXT(cluster.numVertices, cluster.numTriangles);
  
  vec3s_in oPositions = vec3s_in(instance.positions);
  vec3s_in oNormals   = vec3s_in(instance.normals);
  
#if !CLUSTER_DEDICATED_VERTICES
  // the global vertex indices used within this cluster
  uints_in  localVertices  = uints_in(instance.clusterLocalVertices);
#endif
  // the local triangle indices used within this cluster
  uint8s_in localTriangles = uint8s_in(instance.clusterLocalTriangles);
  
  mat4 worldMatrix   = instance.worldMatrix;
  mat3 worldMatrixIT = transpose(inverse(mat3(worldMatrix)));

  // We unroll to force loading vertices & triangles in advance.
  // This reduces latency / dependent loads in the shader. 
  // Because the cluster generators will mostly saturate packing
  // triangles and vertices in a cluster, we normally hardly waste
  // any loading.

  [[unroll]]
  for (uint i = 0; i < uint(MESHLET_VERTEX_ITERATIONS); i++)
  {
    uint vert = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    // Clamp the load because we force processing over max vertices.
    // An alternative to clamping the load index would be to just over-allocate a bit
    // space in the appropriate buffers so we can always do a load operation.
    uint vertLoad = min(vert, vertMax);

#if CLUSTER_DEDICATED_VERTICES
    uint vertexIndex = vertLoad + cluster.firstLocalVertex;
#else
    // Convert the per-cluster vertex into the shared geometry wide vertex index.
    // This allows re-use of vertices across clusters.
    uint vertexIndex = localVertices.d[vertLoad + cluster.firstLocalVertex];
#endif

    vec3 oPos = oPositions.d[vertexIndex];
    vec4 wPos = worldMatrix * vec4(oPos,1.0f);
    
    vec3 oNormal = oNormals.d[vertexIndex];
    
    if (vert <= vertMax) {
      gl_MeshVerticesEXT[vert].gl_Position = view.viewProjMatrix * wPos;
      OUT[vert].wPos    = wPos.xyz;
      OUT[vert].wNormal  = normalize(worldMatrixIT * oNormal);
      OUT[vert].clusterID = gl_WorkGroupID.x;
    }
  }
  
  [[unroll]]
  for (uint i = 0; i < uint(MESHLET_TRIANGLE_ITERATIONS); i++)
  {
    uint tri = gl_LocalInvocationID.x + i * MESHSHADER_WORKGROUP_SIZE;
    uint triLoad = min(tri, triMax);
    
    uvec3 indices = uvec3(localTriangles.d[cluster.firstLocalTriangle + triLoad * 3 + 0],
                          localTriangles.d[cluster.firstLocalTriangle + triLoad * 3 + 1],
                          localTriangles.d[cluster.firstLocalTriangle + triLoad * 3 + 2]);

     if (tri <= triMax) {
        gl_PrimitiveTriangleIndicesEXT[tri] = indices;
        gl_MeshPrimitivesEXT[tri].gl_PrimitiveID = int(tri);
     }
  }
}