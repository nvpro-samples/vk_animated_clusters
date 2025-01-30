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

#ifndef _SHADERIO_H_
#define _SHADERIO_H_

#include "dh_sky.h"

//////////////////////////////////////////////////////////////////////////

#define BBOXES_PER_MESHLET 8

//////////////////////////////////////////////////////////////////////////

#define STATISTICS_WORKGROUP_SIZE 64
#define ANIMATION_WORKGROUP_SIZE 256

//////////////////////////////////////////////////////////////////////////

#define VISUALIZE_NONE 0
#define VISUALIZE_CLUSTER 1
#define VISUALIZE_TRIANGLES 2

//////////////////////////////////////////////////////////////////////////

#define BINDINGS_FRAME_UBO 0
#define BINDINGS_READBACK_SSBO 1
#define BINDINGS_RENDERINSTANCES_SSBO 2
#define BINDINGS_TLAS 3
#define BINDINGS_RENDER_TARGET 4

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
namespace shaderio {
using namespace glm;
using namespace nvvkhl_shaders;
#define BUFFER_REF(typ) uint64_t
#define BUFFER_REF_DECLARE_ARRAY(refname, typ, keywords, alignment)                                                    \
  static_assert(alignof(typ) == alignment || (alignment > alignof(typ) && ((alignment % alignof(typ)) == 0)),          \
                "Alignment incompatible: " #refname)
#else

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_atomic_int64 : enable

#define BUFFER_REF(refname) refname
#define BUFFER_REF_DECLARE_ARRAY(refname, typ, keywords, alignment)                                                    \
  layout(buffer_reference, buffer_reference_align = alignment, scalar) keywords buffer refname                         \
  {                                                                                                                    \
    typ d[];                                                                                                           \
  };

BUFFER_REF_DECLARE_ARRAY(uint8s_in, uint8_t, readonly, 4);
BUFFER_REF_DECLARE_ARRAY(uints_in, uint, readonly, 4);
BUFFER_REF_DECLARE_ARRAY(uint64s_inout, uint64_t, , 8);
BUFFER_REF_DECLARE_ARRAY(uvec3s_in, uvec3, readonly, 4);
BUFFER_REF_DECLARE_ARRAY(vec3s_in, vec3, readonly, 4);
BUFFER_REF_DECLARE_ARRAY(vec3s_inout, vec3, , 4);
#endif

struct BBox
{
  vec3 lo;
  vec3 hi;
};

struct Cluster
{
  uint16_t numVertices;
  uint16_t numTriangles;
  uint32_t firstTriangle;
  uint32_t firstLocalVertex;
  uint32_t firstLocalTriangle;
};
BUFFER_REF_DECLARE_ARRAY(Clusters_in, Cluster, readonly, 16);

struct RenderInstance
{
  mat4 worldMatrix;

  uint32_t numTriangles;
  uint32_t numVertices;
  uint32_t numClusters;
  uint32_t geometryID;

  // animated
  uint64_t positions;
  uint64_t normals;

  // original
  uint64_t triangles;
  uint64_t clusters;
  uint64_t clusterLocalVertices;
  uint64_t clusterLocalTriangles;
  uint64_t clusterBboxes;
  uint64_t originalPositions;
};

struct FrameConstants
{
  mat4 projMatrix;
  mat4 projMatrixI;

  mat4 viewProjMatrix;
  mat4 viewProjMatrixI;
  mat4 viewMatrix;
  mat4 viewMatrixI;
  vec4 viewPos;
  vec4 viewDir;
  vec4 viewPlane;

  ivec2 viewport;
  vec2  viewportf;

  vec2 viewPixelSize;
  vec2 viewClipSize;

  vec3 wLightPos;
  uint _pad1;

  vec2  _padShadow;
  float lightMixer;
  uint  doShadow;

  vec3 wUpDir;
  uint visualize;

  vec4 bgColor;

  float   lodScale;
  float   animationState;
  float   ambientOcclusionRadius;
  int32_t ambientOcclusionRays;

  int32_t animationRippleEnabled;
  float   animationRippleFrequency;
  float   animationRippleAmplitude;
  float   animationRippleSpeed;

  int32_t animationTwistEnabled;
  float   animationTwistSpeed;
  float   animationTwistMaxAngle;
  float   sceneSize;


  uint  doAnimation;
  uint  _pad;
  float nearPlane;
  float farPlane;

  vec4 hizSizeFactors;
  vec4 nearSizeFactors;

  float hizSizeMax;
  int   facetShading;
  int   supersample;
  uint  colorXor;

  uint                dbgUint;
  float               dbgFloat;
  float               time;
  uint                frame;

  uvec2 mousePosition;
  float wireThickness;
  float wireSmoothing;

  vec3 wireColor;
  uint wireStipple;

  vec3  wireBackfaceColor;
  float wireStippleRepeats;

  float wireStippleLength;
  uint  doWireframe;
  uint  visFilterInstanceID;
  uint  visFilterClusterID;

  SimpleSkyParameters skyParams;
};

struct Readback
{
  uint64_t clustersSize;
  uint64_t blasesSize;

#ifndef __cplusplus
  uint64_t clusterTriangleId;
  uint64_t instanceId;
#else
  uint32_t clusterTriangleId;
  uint32_t _packedDepth0;

  uint32_t instanceId;
  uint32_t _packedDepth1;
#endif

  int  debugI;
  uint debugUI;
  uint debugA[32];
  uint debugB[32];
  uint debugC[32];
};

struct AnimationConstants
{
  uint64_t renderInstances;

  uint32_t instanceIndex;
  float    animationState;

  uint32_t rippleEnabled;
  float    rippleFrequency;
  float    rippleAmplitude;
  float    rippleSpeed;

  uint32_t twistEnabled;
  float    twistSpeed;
  float    twistMaxAngle;
  float    geometrySize;
};

struct StatisticsConstants
{
  BUFFER_REF(uints_in) sizes;
  BUFFER_REF(uint64s_inout) sum;
  uint32_t count;
};

struct RayPayload
{
  vec4 color;
};

#ifdef __cplusplus
}
#endif

#endif
