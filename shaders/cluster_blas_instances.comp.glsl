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

#include "shaderio.h"

layout(local_size_x = CLUSTER_BLAS_WORKGROUP_SIZE) in;
layout(push_constant, scalar) uniform pushConstant {
  ClusterBlasConstants push;
};

void main()
{
  uint idx = gl_GlobalInvocationID.x;

  // update ray tracing blas address prior tlas update
  if (idx < push.instanceCount)
  {
    push.rayInstances.d[idx].accelerationStructureReference = push.blasAddresses.d[ push.animated != 0 ? idx : push.instances.d[idx].geometryID ];
  }

  // for statistics we sum blas/cluster sizes
  if (idx < push.sumCount)
  {
    atomicAdd(push.sum.d[0], uint64_t(push.sizes.d[idx]));
  }
}