/*
* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

#version 460

void main()
{
  vec2 pos;
  pos.x = float(gl_VertexIndex >> 1);
  pos.y = float(gl_VertexIndex & 1);
  
  gl_Position = vec4(pos * 4 - 1, 1.0, 1.0);
}