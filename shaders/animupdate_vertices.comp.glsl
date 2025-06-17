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

#include "shaderio.h"
#define M_PI 3.14159265358979323f

layout(push_constant) uniform animationConstantsPush
{
  AnimationConstants constants;
};


layout(buffer_reference, buffer_reference_align = 4, scalar) buffer F32Buffer
{
  float v[];
};

layout(local_size_x = ANIMATION_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

mat3 rotationMatrix(vec3 axis, float angle)
{
  axis     = normalize(axis);
  float s  = sin(angle);
  float c  = cos(angle);
  float oc = 1.0 - c;

  return mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
              oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
              oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}


void main()
{
  uint32_t index = gl_GlobalInvocationID.x;

  RenderInstance instance = RenderInstances_in(constants.renderInstances).d[constants.instanceIndex];


  if(index >= instance.numVertices)
  {
    return;
  }
  
  float seed = float(instance.positions) / float(~0u);
  vec3  originalVertex;
  for(uint32_t i = 0; i < 3; i++)
  {

    float coord                                  = F32Buffer(instance.originalPositions).v[3 * index + i];
    originalVertex[i]                            = coord;
    F32Buffer(instance.normals).v[3 * index + i] = 0.f;
  }

  vec3 newVertex = originalVertex;

  if(constants.rippleEnabled != 0 && constants.animationState != 0.f)
  {

    float maxCoord = max(abs(originalVertex.x), max(abs(originalVertex.y), abs(originalVertex.z)));

    float frequency = constants.rippleFrequency / constants.geometrySize;

    vec3 wave = vec3(sin(maxCoord * frequency + seed + constants.animationState * constants.rippleSpeed),
                     cos(maxCoord * frequency * 3 + seed + constants.animationState * constants.rippleSpeed),
                     sin(maxCoord * frequency * 1.2f + seed + constants.animationState * constants.rippleSpeed));
    newVertex += (normalize(originalVertex.zyx)) * (constants.rippleAmplitude * constants.geometrySize * wave);
  }

  if(constants.twistEnabled != 0 && constants.animationState != 0.f)
  {
    float time  = constants.animationState * constants.twistSpeed;
    float stage = mod(time, 3.f);
    vec3  axis;
    for(uint32_t i = 0; i < 3; i++)
    {
      if(stage >= i && stage <= i + 1)
      {
        axis[i] = 1;
      }
      else
      {
        axis[i] = 0;
      }
    }

    float angle = (sin(time * 2.f * M_PI) * length(originalVertex / (constants.geometrySize * .5f)) * constants.twistMaxAngle);

    mat3 rotation = rotationMatrix(axis, angle);
    newVertex     = rotation * newVertex;
  }


  for(uint32_t i = 0; i < 3; i++)
  {
    F32Buffer(instance.positions).v[3 * index + i] = newVertex[i];
  }
}