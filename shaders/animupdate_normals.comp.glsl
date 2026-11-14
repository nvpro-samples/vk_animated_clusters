/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_float : enable

#include "shaderio.h"

layout(push_constant) uniform animationConstantsPush
{
  AnimationConstants constants;
};

layout(buffer_reference, scalar) readonly buffer U32Buffer
{
  uint32_t i[];
};

layout(buffer_reference, scalar) buffer F32Buffer
{
  float v[];
};

layout(local_size_x = ANIMATION_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
  uint32_t index = gl_GlobalInvocationID.x;

  RenderInstance instance = constants.instances.d[constants.instanceIndex];

  if(index >= instance.numTriangles)
  {
    return;
  }

  vec3 vertices[3];
  for(uint32_t i = 0; i < 3; i++)
  {
    uint32_t vertexIndex = U32Buffer(instance.triangles).i[3 * index + i];
    for(uint32_t axis = 0; axis < 3; axis++)
    {
      vertices[i][axis] = F32Buffer(instance.positions).v[3 * vertexIndex + axis];
    }
  }

  vec3 e0 = vertices[1] - vertices[0];
  vec3 e1 = vertices[2] - vertices[0];

  vec3 n = normalize(cross(e0, e1));

  for(uint32_t i = 0; i < 3; i++)
  {
    uint32_t vertexIndex = U32Buffer(instance.triangles).i[3 * index + i];

    for(uint32_t axis = 0; axis < 3; axis++)
    {
      atomicAdd(F32Buffer(instance.normals).v[3 * vertexIndex + axis], n[axis]);
    }
  }
}