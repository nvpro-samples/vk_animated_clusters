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

#include <nvcluster/nvcluster.h>

#include "resources.hpp"

namespace animatedclusters {
struct SceneConfig
{
  uint32_t         clusterVertices           = 64;
  uint32_t         clusterTriangles          = 64;
  nvcluster_Config clusterNvConfig           = {};
  float            clusterMeshoptSpatialFill = 0.5f;

  bool clusterDedicatedVertices = false;
  bool clusterStripify          = true;
  bool clusterNvLibrary         = false;

  // Influence the number of geometries that can be processed in parallel.
  // Percentage of threads of maximum hardware concurrency
  float processingThreadsPct = 0.5;
};

class Scene
{
public:
  struct Instance
  {
    glm::mat4      matrix;
    shaderio::BBox bbox;
    uint32_t       geometryID = ~0U;
    uint32_t       materialID = 0;
  };

  struct Geometry
  {
    uint32_t numTriangles{};
    uint32_t numVertices{};
    uint32_t numClusters{};
    uint32_t numClusterVertices{};

    shaderio::BBox bbox;

    std::vector<glm::vec3>  positions;
    std::vector<glm::uvec3> triangles;
    std::vector<uint8_t>    clusterLocalTriangles;
    std::vector<uint32_t>   clusterLocalVertices;

    std::vector<shaderio::Cluster> clusters;
    std::vector<shaderio::BBox>    clusterBboxes;

    nvvk::Buffer positionsBuffer;
    nvvk::Buffer trianglesBuffer;
    nvvk::Buffer clustersBuffer;
    nvvk::Buffer clusterLocalTrianglesBuffer;
    nvvk::Buffer clusterLocalVerticesBuffer;
    nvvk::Buffer clusterBboxesBuffer;
  };

  struct Camera
  {
    glm::mat4 worldMatrix{1};
    glm::vec3 eye{0, 0, 0};
    glm::vec3 center{0, 0, 0};
    glm::vec3 up{0, 1, 0};
    float     fovy;
  };

  bool init(const std::filesystem::path& filename, const SceneConfig& config, Resources& res);
  void deinit(Resources& res);

  SceneConfig m_config;

  shaderio::BBox m_bbox;

  std::vector<Instance> m_instances;
  std::vector<Geometry> m_geometries;
  std::vector<Camera>   m_cameras;

  size_t   m_sceneClusterMemBytes          = 0;
  size_t   m_sceneTriangleMemBytes         = 0;
  uint32_t m_maxClusterTriangles           = 0;
  uint32_t m_maxClusterVertices            = 0;
  uint32_t m_maxPerGeometryClusters        = 0;
  uint32_t m_maxPerGeometryTriangles       = 0;
  uint32_t m_maxPerGeometryVertices        = 0;
  uint32_t m_maxPerGeometryClusterVertices = 0;
  uint32_t m_numClusters                   = 0;
  uint32_t m_numTriangles                  = 0;

  std::vector<uint32_t> m_clusterTriangleHistogram;
  std::vector<uint32_t> m_clusterVertexHistogram;
  uint32_t              m_clusterTriangleHistogramMax;
  uint32_t              m_clusterVertexHistogramMax;


private:
  struct ProcessingInfo
  {
    // how we perform multi-threading:
    // - either over geometries (outer loop)
    // - or within a geometry (inner loops)

    nvcluster_Context clusterContext{};

    uint32_t numPoolThreadsOriginal = 1;
    uint32_t numPoolThreads         = 1;

    uint32_t numOuterThreads = 1;
    uint32_t numInnerThreads = 1;

    size_t geometryCount = 0;

    std::mutex processOnlySaveMutex;

    // some stats

    std::atomic_uint64_t numTotalTriangles = 0;
    std::atomic_uint64_t numTotalStrips    = 0;

    std::mutex statsMutex;

    // logging progress

    uint32_t   progressLastPercentage      = 0;
    uint32_t   progressGeometriesCompleted = 0;
    std::mutex progressMutex;

    nvutils::PerformanceTimer clock;
    double                    startTime = 0;

    void init(float pct);
    void setupParallelism(size_t geometryCount_);
    void deinit();

    void logBegin();
    void logCompletedGeometry();
    void logEnd();
  };

  bool loadGLTF(ProcessingInfo& processingInfo, const std::filesystem::path& filename);

  void processGeometry(ProcessingInfo& processingInfo, Geometry& geometry);
  void buildGeometryClusters(ProcessingInfo& processingInfo, Geometry& geometry);
  void buildGeometryClusterBboxes(ProcessingInfo& processingInfo, Geometry& geometry);
  void buildGeometryClusterStrips(ProcessingInfo& processingInfo, Geometry& geometry);
  void buildGeometryClusterVertices(ProcessingInfo& processingInfo, Geometry& geometry);
  void rebuildGeometryTriangles(ProcessingInfo& processingInfo, Geometry& geometry);

  void computeInstanceBBoxes();
  void initGpuBuffers(Resources& res);
};
}  // namespace animatedclusters
