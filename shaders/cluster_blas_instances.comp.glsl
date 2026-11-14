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