/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

void main()
{
  vec2 pos;
  pos.x = float(gl_VertexIndex >> 1);
  pos.y = float(gl_VertexIndex & 1);
  
  gl_Position = vec4(pos * 4 - 1, 0.0, 1.0);
}