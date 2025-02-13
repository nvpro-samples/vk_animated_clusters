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
#pragma once

#include <vector>

#include "resources.hpp"


namespace animatedclusters {
struct SceneConfig
{
  uint32_t clusterVertices  = 64;
  uint32_t clusterTriangles = 64;
  // 0 disables
  float clusterNvGraphWeight = 0.0;
  // Cost penalty for under-filling clusters
  float clusterNvUnderfill = 1.0f;
  // Cost penalty for overlapping bounding boxes
  float clusterNvOverlap = 0.5f;

  bool clusterDedicatedVertices = false;
  bool clusterStripify          = true;
  bool clusterNvLibrary         = true;
};

class Scene
{
public:
  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
  };

  struct Geometry
  {
    uint32_t numTriangles;
    uint32_t numVertices;
    uint32_t numClusters;

    shaderio::BBox bbox;

    std::vector<glm::vec3>  positions;
    std::vector<glm::uvec3> triangles;
    std::vector<uint8_t>    clusterLocalTriangles;
    std::vector<uint32_t>   clusterLocalVertices;

    std::vector<shaderio::Cluster> clusters;
    std::vector<shaderio::BBox>    clusterBboxes;

    RBuffer positionsBuffer;
    RBuffer trianglesBuffer;
    RBuffer clustersBuffer;
    RBuffer clusterLocalTrianglesBuffer;
    RBuffer clusterLocalVerticesBuffer;
    RBuffer clusterBboxesBuffer;
  };

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  bool init(const char* filename, Resources& res, const SceneConfig& config);
  void deinit(Resources& res);

  SceneConfig m_config;

  shaderio::BBox m_bbox;

  std::vector<Instance> m_instances;
  std::vector<Geometry> m_geometries;
  std::vector<Camera>   m_cameras;

  size_t                m_sceneClusterMemBytes          = 0;
  size_t                m_sceneTriangleMemBytes         = 0;
  uint32_t              m_maxPerGeometryClusters        = 0;
  uint32_t              m_maxPerGeometryTriangles       = 0;
  uint32_t              m_maxPerGeometryVertices        = 0;
  uint32_t              m_maxPerGeometryClusterVertices = 0;
  uint32_t              m_numClusters                   = 0;
  uint32_t              m_numTriangles                  = 0;
  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;

  uint32_t m_clusterTriangleHistogramMax;
  uint32_t m_clusterVertexHistogramMax;


private:
  bool loadGLTF(const char* filename);
  bool buildClusters();
  void computeBBoxes();
  void upload(Resources& res);
};
}  // namespace animatedclusters
