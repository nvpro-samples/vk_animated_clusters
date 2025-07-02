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

layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer BBoxes_in
{
  BBox d[];
};

////////////////////////////////////////////

layout(location=0) out Interpolants {
  flat uint clusterID;
} OUT[];

////////////////////////////////////////////

#define MESH_WORKGROUP_SIZE  32

#define BOX_VERTICES     8
#define BOX_LINES        12
#define BOX_LINE_THREADS 4

layout(local_size_x=MESH_WORKGROUP_SIZE) in;
layout(max_vertices=BBOXES_PER_MESHLET * BOX_VERTICES, max_primitives=BBOXES_PER_MESHLET * BOX_LINES) out;
layout(lines) out;

////////////////////////////////////////////

void main()
{
  RenderInstance instance = instances[push.instanceID];
  
  BBoxes_in bboxes = BBoxes_in(instance.clusterBboxes);
  
  uint baseID   = gl_WorkGroupID.x * BBOXES_PER_MESHLET;  
  uint numBoxes = min(instance.numClusters, baseID + BBOXES_PER_MESHLET) - baseID;
  
  SetMeshOutputsEXT(numBoxes * 8, numBoxes * 12);
  
  const uint vertexRuns = ((BBOXES_PER_MESHLET * BOX_VERTICES) + MESH_WORKGROUP_SIZE-1) / MESH_WORKGROUP_SIZE;
  
  [[unroll]]
  for (uint32_t run = 0; run < vertexRuns; run++)
  {
    uint vert   = gl_LocalInvocationID.x + run * MESH_WORKGROUP_SIZE;
    uint box    = vert / BOX_VERTICES;
    uint corner = vert % BOX_VERTICES;
    
    uint boxLoad = min(box,numBoxes-1);
    
    BBox bbox = bboxes.d[boxLoad + baseID];
    
    bvec3 weight   = bvec3((corner & 1) != 0, (corner & 2) != 0, (corner & 4) != 0);
    vec3 cornerPos = mix(bbox.lo, bbox.hi, weight);
    
    if (box < numBoxes)
    {
      gl_MeshVerticesEXT[vert].gl_Position = view.viewProjMatrix * (instance.worldMatrix * vec4(cornerPos,1));
      OUT[vert].clusterID = baseID + box;
    }
  }
  
  
  {
    uvec2 boxIndices[4] = uvec2[4](
      uvec2(0,1),uvec2(1,3),uvec2(3,2),uvec2(2,0)
    );
  
    uint subID = gl_LocalInvocationID.x & (BOX_LINE_THREADS-1);
    uint box   = gl_LocalInvocationID.x / BOX_LINE_THREADS;
  
    uvec2 circle = boxIndices[subID];
    
    if (box < numBoxes)
    {  
      // lower
      gl_PrimitiveLineIndicesEXT[box * 12 + subID + 0] = circle + box * BOX_VERTICES;
      // upper
      gl_PrimitiveLineIndicesEXT[box * 12 + subID + 4] = circle + 4 + box * BOX_VERTICES;
      // connectors
      gl_PrimitiveLineIndicesEXT[box * 12 + subID + 8] = uvec2(subID, subID + 4) + box * BOX_VERTICES;;
    }
  }
}
