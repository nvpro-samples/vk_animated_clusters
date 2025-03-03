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

#include <meshoptimizer.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <nvh/parallel_work.hpp>

#include "scene.hpp"

namespace animatedclusters {

bool Scene::init(const char* filename, Resources& res, const SceneConfig& config)
{
  m_config = config;

  if(!loadGLTF(filename))
  {
    return false;
  }

  computeBBoxes();

  if(!buildClusters())
  {
    return false;
  }

  for(auto& geom : m_geometries)
  {
    m_maxPerGeometryTriangles = std::max(m_maxPerGeometryTriangles, geom.numTriangles);
    m_maxPerGeometryVertices  = std::max(m_maxPerGeometryVertices, geom.numVertices);
    m_numTriangles += geom.numTriangles;
  }

  upload(res);

  return true;
}

void Scene::deinit(Resources& res)
{
  for(auto& geom : m_geometries)
  {
    res.destroy(geom.positionsBuffer);
    res.destroy(geom.trianglesBuffer);
    res.destroy(geom.clustersBuffer);
    res.destroy(geom.clusterLocalTrianglesBuffer);
    res.destroy(geom.clusterLocalVerticesBuffer);
    res.destroy(geom.clusterBboxesBuffer);
  }
}

void Scene::computeBBoxes()
{
  m_bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};

  for(auto& instance : m_instances)
  {
    assert(instance.geometryID <= m_geometries.size());

    const Geometry& geom = m_geometries[instance.geometryID];

    instance.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};

    for(uint32_t v = 0; v < 8; v++)
    {
      bool x = (v & 1) != 0;
      bool y = (v & 2) != 0;
      bool z = (v & 4) != 0;

      glm::bvec3 weight(x, y, z);
      glm::vec3  corner = glm::mix(geom.bbox.lo, geom.bbox.hi, weight);
      corner            = instance.matrix * glm::vec4(corner, 1.0f);
      instance.bbox.lo  = glm::min(instance.bbox.lo, corner);
      instance.bbox.hi  = glm::max(instance.bbox.hi, corner);
    }

    m_bbox.lo = glm::min(m_bbox.lo, instance.bbox.lo);
    m_bbox.hi = glm::max(m_bbox.hi, instance.bbox.hi);
  }
}

void Scene::upload(Resources& res)
{
  m_sceneClusterMemBytes  = 0;
  m_sceneTriangleMemBytes = 0;

  Resources::BatchedUploader uploader(res);

  // not exactly efficient upload ;)
  for(auto& geom : m_geometries)
  {
    if(geom.positions.size())
    {
      geom.positionsBuffer = res.createBuffer(sizeof(glm::vec3) * geom.positions.size(),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

      uploader.uploadBuffer(geom.positionsBuffer, geom.positions.data());

      m_sceneClusterMemBytes += geom.positionsBuffer.info.range;
      m_sceneTriangleMemBytes += geom.positionsBuffer.info.range;
    }
    if(geom.triangles.size())
    {
      geom.trianglesBuffer = res.createBuffer(sizeof(glm::uvec3) * geom.triangles.size(),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                                  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      uploader.uploadBuffer(geom.trianglesBuffer, geom.triangles.data());

      // animation still needs triangles for normals, even if clusters are used
      m_sceneClusterMemBytes += geom.trianglesBuffer.info.range;
      m_sceneTriangleMemBytes += geom.trianglesBuffer.info.range;
    }
    if(geom.clusters.size())
    {
      geom.clustersBuffer = res.createBuffer(sizeof(shaderio::Cluster) * geom.clusters.size(),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                 | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      uploader.uploadBuffer(geom.clustersBuffer, geom.clusters.data());

      m_sceneClusterMemBytes += geom.clustersBuffer.info.range;
    }
    if(geom.clusterLocalTriangles.size())
    {
      geom.clusterLocalTrianglesBuffer =
          res.createBuffer(sizeof(uint8_t) * geom.clusterLocalTriangles.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      uploader.uploadBuffer(geom.clusterLocalTrianglesBuffer, geom.clusterLocalTriangles.data());

      m_sceneClusterMemBytes += geom.clusterLocalTrianglesBuffer.info.range;
    }
    if(geom.clusterLocalVertices.size() && !m_config.clusterDedicatedVertices)
    {
      geom.clusterLocalVerticesBuffer =
          res.createBuffer(sizeof(uint32_t) * geom.clusterLocalVertices.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      uploader.uploadBuffer(geom.clusterLocalVerticesBuffer, geom.clusterLocalVertices.data());

      m_sceneClusterMemBytes += geom.clusterLocalVerticesBuffer.info.range;
    }
    if(geom.clusterBboxes.size())
    {
      geom.clusterBboxesBuffer =
          res.createBuffer(sizeof(shaderio::BBox) * geom.clusterBboxes.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      uploader.uploadBuffer(geom.clusterBboxesBuffer, geom.clusterBboxes.data());

      m_sceneClusterMemBytes += geom.clusterBboxesBuffer.info.range;
    }
  }

  uploader.flush();
}

bool Scene::buildClusters()
{
  nvh::Profiler::Clock clock;
  double               startTime = clock.getMicroSeconds();

  uint32_t numThreads = nvh::get_thread_pool().get_thread_count();

  m_clusterTriangleHistogram.resize(m_config.clusterTriangles + 1, 0);
  m_clusterVertexHistogram.resize((m_config.clusterTriangles * 3) + 1, 0);

  nvcluster::Context           nvclusterContext;
  nvcluster::ContextCreateInfo nvclusterContextCreateInfo;
  nvcluster::Result            nvclusterResult = nvclusterCreateContext(&nvclusterContextCreateInfo, &nvclusterContext);

  if(m_config.clusterNvLibrary)
  {
    // the cluster library doesn't enforce limits
    m_config.clusterVertices = 0;
  }

  std::vector<nvcluster::AABB> triangleAABBs;
  std::vector<glm::vec3>       triangleCentroids;

  std::vector<nvcluster::Range> vertexTriangleRanges;
  std::vector<uint32_t>         vertexTriangles;

  std::vector<nvcluster::Range> triangleConnectionRanges;
  std::vector<float>            triangleConnectionWeights;
  std::vector<uint32_t>         triangleConnectionTargets;

  std::vector<uint32_t> threadCacheEarly(numThreads * 256);

  uint64_t numTotalTriangles = 0;
  uint64_t numTotalStrips    = 0;

  for(size_t g = 0; g < m_geometries.size(); g++)
  {
    Geometry& geom = m_geometries[g];

    if(m_config.clusterNvLibrary)
    {
      triangleAABBs.resize(geom.numTriangles);
      triangleCentroids.resize(geom.numTriangles);
      triangleConnectionRanges.resize(geom.numTriangles);
      vertexTriangleRanges.clear();
      vertexTriangleRanges.resize(geom.numVertices + 1, {0, 0});

      static_assert(std::atomic_uint32_t::is_always_lock_free && sizeof(uint32_t) == sizeof(std::atomic_uint32_t));

      nvh::parallel_batches_indexed(
          geom.numTriangles,
          [&](uint64_t t, uint32_t threadIdx) {
            glm::uvec3 triangleIndices = geom.triangles[t];

            glm::vec3 vertexA = geom.positions[triangleIndices.x];
            glm::vec3 vertexB = geom.positions[triangleIndices.y];
            glm::vec3 vertexC = geom.positions[triangleIndices.z];

            glm::vec3 lo = vertexA;
            glm::vec3 hi = vertexA;

            lo = glm::min(lo, vertexB);
            lo = glm::min(lo, vertexC);

            hi = glm::max(hi, vertexB);
            hi = glm::max(hi, vertexC);

            triangleAABBs[t].bboxMin[0] = lo.x;
            triangleAABBs[t].bboxMin[1] = lo.y;
            triangleAABBs[t].bboxMin[2] = lo.z;
            triangleAABBs[t].bboxMax[0] = hi.x;
            triangleAABBs[t].bboxMax[1] = hi.y;
            triangleAABBs[t].bboxMax[2] = hi.z;

            triangleCentroids[t] = (lo + hi) * 0.5f;

            reinterpret_cast<std::atomic_uint32_t&>(vertexTriangleRanges[triangleIndices.x].count)++;
            reinterpret_cast<std::atomic_uint32_t&>(vertexTriangleRanges[triangleIndices.y].count)++;
            reinterpret_cast<std::atomic_uint32_t&>(vertexTriangleRanges[triangleIndices.z].count)++;
          },
          numThreads);

      // build graph
      if(m_config.clusterNvGraphWeight)
      {
        // compute triangle worst-case connections
        nvh::parallel_batches_indexed(
            geom.numTriangles,
            [&](uint64_t idx, uint32_t threadIdx) {
              glm::uvec3 triangleIndices = geom.triangles[idx];
              // subtract ourselves from each list
              // note triangles with shared edges will still appear in multiple vertex lists
              // hence this is conservative
              triangleConnectionRanges[idx].count = vertexTriangleRanges[triangleIndices.x].count
                                                    + vertexTriangleRanges[triangleIndices.y].count
                                                    + vertexTriangleRanges[triangleIndices.z].count - 3;
            },
            numThreads);

        uint32_t offsetVertexTriangles = 0;
        for(uint32_t v = 0; v < geom.numVertices; v++)
        {
          vertexTriangleRanges[v].offset = offsetVertexTriangles;
          offsetVertexTriangles += vertexTriangleRanges[v].count;
          vertexTriangleRanges[v].count = 0;
        }
        // just used as terminator to detect out of bounds
        vertexTriangleRanges[geom.numVertices].offset = offsetVertexTriangles;

        vertexTriangles.resize(offsetVertexTriangles);

        uint32_t offsetTriangleConnections = 0;
        for(uint32_t t = 0; t < geom.numTriangles; t++)
        {
          // setup offset of triangle connections based on conservative estimate
          triangleConnectionRanges[t].offset = offsetTriangleConnections;
          offsetTriangleConnections += triangleConnectionRanges[t].count;

          // append triangle to vertex lists
          glm::uvec3 triangleIndices = geom.triangles[t];
          for(uint32_t k = 0; k < 3; k++)
          {
            uint32_t          vertexIndex                             = triangleIndices[k];
            nvcluster::Range& vertexRange                             = vertexTriangleRanges[vertexIndex];
            vertexTriangles[vertexRange.offset + vertexRange.count++] = t;
#ifdef _DEBUG
            uint32_t maxCount = vertexTriangleRanges[vertexIndex + 1].offset - vertexRange.offset;
            assert(vertexRange.count <= maxCount);
#endif
          }
        }

        triangleConnectionTargets.resize(offsetTriangleConnections);
        triangleConnectionWeights.resize(offsetTriangleConnections);

        // compute accurate triangle connections
        nvh::parallel_batches_indexed(
            geom.numTriangles,
            [&](uint64_t t, uint32_t threadIdx) {
              glm::uvec3        triangleIndices = geom.triangles[t];
              nvcluster::Range& triangleRange   = triangleConnectionRanges[t];

              uint32_t* connectionTargets = &triangleConnectionTargets[triangleRange.offset];
              float*    connectionWeights = &triangleConnectionWeights[triangleRange.offset];
              uint32_t* connectionEarly   = &threadCacheEarly[threadIdx * 256];
              memset(connectionEarly, ~0, sizeof(uint32_t) * 256);

              uint32_t newCount = 0;
              for(uint32_t k = 0; k < 3; k++)
              {
                uint32_t         vertexIndex            = triangleIndices[k];
                nvcluster::Range vertexRange            = vertexTriangleRanges[vertexIndex];
                const uint32_t*  currentVertexTriangles = &vertexTriangles[vertexRange.offset];
                for(uint32_t vt = 0; vt < vertexRange.count; vt++)
                {
                  uint32_t vertexTriangle = currentVertexTriangles[vt];

                  // ignore ourself
                  // or if already in list (opportunistic check)
                  if(vertexTriangle == uint32_t(t) || (connectionEarly[vertexTriangle & 0xFF] == vertexTriangle))
                    continue;

                  // otherwise search list in detail
                  bool found = false;
                  for(uint32_t t = 0; t < newCount; t++)
                  {
                    if(vertexTriangle == connectionTargets[t])
                    {
                      found = true;
                      break;
                    }
                  }

                  if(!found)
                  {
                    connectionEarly[vertexTriangle & 0xFF] = vertexTriangle;
                    connectionTargets[newCount]            = vertexTriangle;
                    connectionWeights[newCount]            = m_config.clusterNvGraphWeight;
                    newCount++;
                  }
                }
              }

              assert(newCount <= triangleRange.count);
              triangleRange.count = newCount;
            },
            numThreads);
      }

      nvcluster::SpatialElements spatial;
      spatial.boundingBoxes = triangleAABBs.data();
      spatial.centroids     = reinterpret_cast<const float*>(triangleCentroids.data());
      spatial.elementCount  = geom.numTriangles;

      nvcluster::Graph graph;
      if(m_config.clusterNvGraphWeight)
      {
        graph.connectionCount   = uint32_t(triangleConnectionTargets.size());
        graph.connectionTargets = triangleConnectionTargets.data();
        graph.connectionWeights = triangleConnectionWeights.data();
        graph.nodeCount         = geom.numTriangles;
        graph.nodes             = triangleConnectionRanges.data();
      }

      nvcluster::Input input;
      input.config.costUnderfill  = m_config.clusterNvUnderfill;
      input.config.costOverlap    = m_config.clusterNvOverlap;
      input.config.maxClusterSize = m_config.clusterTriangles;
      input.spatialElements       = &spatial;
      input.graph                 = m_config.clusterNvGraphWeight ? &graph : nullptr;

      nvcluster::ClusterStorage storage;
      nvclusterResult = nvcluster::generateClusters(nvclusterContext, input, storage);

      size_t numClusters = storage.clusterRanges.size();
      geom.numClusters   = uint32_t(numClusters);

      if(numClusters)
      {
        geom.clusterLocalTriangles.resize(geom.numTriangles * 3);
        geom.clusterLocalVertices.resize(geom.numTriangles * 3);
        geom.clusters.resize(numClusters);

        // linearize triangle offsets
        uint32_t firstLocalTriangleOffset = 0;
        for(size_t c = 0; c < numClusters; c++)
        {
          shaderio::Cluster& cluster = geom.clusters[c];
          cluster.numTriangles       = storage.clusterRanges[c].count;
          cluster.firstLocalTriangle = firstLocalTriangleOffset;
          firstLocalTriangleOffset += cluster.numTriangles * 3;
        }

        // fill clusters
        nvh::parallel_batches_indexed(
            numClusters,
            [&](uint64_t idx, uint32_t threadIdx) {
              shaderio::Cluster& cluster      = geom.clusters[idx];
              nvcluster::Range   clusterRange = storage.clusterRanges[idx];

              uint8_t*  localTriangles   = &geom.clusterLocalTriangles[cluster.firstLocalTriangle];
              uint32_t* localVertices    = &geom.clusterLocalVertices[cluster.firstLocalTriangle];
              uint32_t* localItems       = &storage.clusterItems[clusterRange.offset];
              uint32_t* vertexCacheEarly = &threadCacheEarly[threadIdx * 256];
              memset(vertexCacheEarly, ~0, sizeof(uint32_t) * 256);

              uint32_t numVertices = 0;
              uint32_t numIndices  = 0;

              for(uint32_t t = 0; t < cluster.numTriangles; t++)
              {
                glm::uvec3 triangleIndices = geom.triangles[localItems[t]];
                for(uint32_t k = 0; k < 3; k++)
                {
                  uint32_t vertexIndex = triangleIndices[k];
                  bool     found       = false;

                  // quick early out
                  // upper bits of full 32-bit vertex index
                  // lower bits of slot in vertex cache
                  uint32_t cacheEarlyValue = vertexCacheEarly[vertexIndex & 0xFF];
                  if((cacheEarlyValue >> 8) == (vertexIndex >> 8))
                  {
                    localTriangles[numIndices++] = uint8_t(cacheEarlyValue & 0xFF);
                    continue;
                  }
                  // otherwise search list in detail
                  for(uint32_t v = 0; v < numVertices; v++)
                  {
                    if(localVertices[v] == vertexIndex)
                    {
                      found                        = true;
                      localTriangles[numIndices++] = uint8_t(v);
                      break;
                    }
                  }

                  if(!found)
                  {
                    vertexCacheEarly[vertexIndex & 0xFF] = (vertexIndex & ~0xFF) | (numVertices & 0xFF);
                    localTriangles[numIndices++]         = numVertices;
                    localVertices[numVertices++]         = vertexIndex;
                  }
                }
              }
              assert(numIndices == cluster.numTriangles * 3);

              assert(numVertices <= 255);
              cluster.numVertices = numVertices;
            },
            numThreads);

        // compact local vertices
        uint32_t writeOffset = 0;
        for(size_t c = 0; c < numClusters; c++)
        {
          shaderio::Cluster& cluster = geom.clusters[c];
          cluster.firstLocalVertex   = writeOffset;

          memmove(&geom.clusterLocalVertices[writeOffset], &geom.clusterLocalVertices[cluster.firstLocalTriangle],
                  sizeof(uint32_t) * cluster.numVertices);

          // cluster library has no way to enforce vertex limit
          m_config.clusterVertices = std::max(uint32_t(cluster.numVertices), m_config.clusterVertices);

          writeOffset += uint32_t(cluster.numVertices);
        }
        geom.clusterLocalVertices.resize(writeOffset);
        geom.clusterLocalVertices.shrink_to_fit();

        if(m_config.clusterVertices > 256)
        {
          LOGE("FATAL ERROR: geometry %zu generated clusters beyond 256 limitation\n", g);
          nvclusterDestroyContext(nvclusterContext);
          return false;
        }
      }
    }
    else
    {
      // first sort for vcache
      std::vector<glm::uvec3> triangles = geom.triangles;
      meshopt_optimizeVertexCache((uint32_t*)geom.triangles.data(), (uint32_t*)triangles.data(), triangles.size() * 3, geom.numVertices);

      // we allow smaller clusters to be generated when that significantly improves their bounds
      size_t minTriangles = (m_config.clusterTriangles / 4) & ~3;

      std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(geom.numTriangles * 3, m_config.clusterVertices, minTriangles));
      geom.clusterLocalTriangles.resize(meshlets.size() * m_config.clusterTriangles * 3);
      geom.clusterLocalVertices.resize(meshlets.size() * m_config.clusterVertices);

      const float coneWeight = -1.f; // use axis aligned metrics
      const float splitFactor = 2.f; // limit disconnected clusters

      size_t numClusters =
          meshopt_buildMeshletsFlex(meshlets.data(), geom.clusterLocalVertices.data(), geom.clusterLocalTriangles.data(),
                                    (uint32_t*)geom.triangles.data(), geom.triangles.size() * 3,
                                    (float*)geom.positions.data(), geom.numVertices, sizeof(glm::vec3),
                                    std::min(255u, m_config.clusterVertices), minTriangles, m_config.clusterTriangles, coneWeight, splitFactor);

      geom.numClusters = uint32_t(numClusters);

      if(geom.numClusters)
      {
        geom.clusters.resize(geom.numClusters);
        geom.clusters.shrink_to_fit();

        for(size_t c = 0; c < numClusters; c++)
        {
          meshopt_Meshlet&   meshlet = meshlets[c];
          shaderio::Cluster& cluster = geom.clusters[c];

          cluster.numTriangles       = meshlet.triangle_count;
          cluster.numVertices        = meshlet.vertex_count;
          cluster.firstLocalTriangle = meshlet.triangle_offset;
          cluster.firstLocalVertex   = meshlet.vertex_offset;
        }
      }
    }

    if(geom.numClusters)
    {
      if(m_config.clusterStripify)
      {
        uint32_t numMaxTriangles  = m_config.clusterTriangles;
        uint32_t perThreadIndices = numMaxTriangles * 3 * 2 + uint32_t(meshopt_stripifyBound(numMaxTriangles * 3));

        std::atomic_uint32_t  numStrips = 0;
        std::vector<uint32_t> threadIndices(numThreads * perThreadIndices);

        nvh::parallel_ranges(
            geom.numClusters,
            [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadIdx) {
              for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
              {
                shaderio::Cluster& cluster = geom.clusters[idx];

                uint32_t* meshletIndices      = &threadIndices[threadIdx * perThreadIndices];
                uint32_t* meshletOptim        = meshletIndices + cluster.numTriangles * 3;
                uint32_t* meshletStripIndices = meshletOptim + cluster.numTriangles * 3;

                // convert u8 to u32
                for(uint32_t i = 0; i < uint32_t(cluster.numTriangles) * 3; i++)
                {
                  meshletIndices[i] = geom.clusterLocalTriangles[cluster.firstLocalTriangle + i];
                }

                meshopt_optimizeVertexCache(meshletOptim, meshletIndices, cluster.numTriangles * 3, cluster.numVertices);
                size_t stripIndexCount =
                    meshopt_stripify(meshletStripIndices, meshletOptim, cluster.numTriangles * 3, cluster.numVertices, ~0);
                size_t newIndexCount = meshopt_unstripify(meshletIndices, meshletStripIndices, stripIndexCount, ~0);

                cluster.numTriangles = uint32_t(newIndexCount / 3);

                for(uint32_t i = 0; i < uint32_t(newIndexCount); i++)
                {
                  geom.clusterLocalTriangles[cluster.firstLocalTriangle + i] = uint8_t(meshletIndices[i]);
                }

                // just for stats
                numStrips++;
                for(uint32_t i = 1; i < uint32_t(cluster.numTriangles); i++)
                {
                  const uint32_t* current = meshletIndices + i * 3;
                  const uint32_t* prev    = meshletIndices + (i - 1) * 3;

                  if(!((current[0] == prev[0] || current[0] == prev[2]) && (current[1] == prev[1] || current[1] == prev[2])))
                    numStrips++;
                }
              }
            },
            numThreads);


        numTotalTriangles += geom.numTriangles;
        numTotalStrips += numStrips;
      }

      if(m_config.clusterDedicatedVertices)
      {
        std::vector<glm::vec3> oldPositions = std::move(geom.positions);
        geom.positions.resize(geom.clusterLocalVertices.size());
        geom.numVertices = uint32_t(geom.positions.size());

        nvh::parallel_ranges(
            geom.numClusters,
            [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadIdx) {
              for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
              {
                shaderio::Cluster& cluster = geom.clusters[idx];

                for(uint32_t v = 0; v < cluster.numVertices; v++)
                {
                  uint32_t newIndex                   = cluster.firstLocalVertex + v;
                  uint32_t oldIndex                   = geom.clusterLocalVertices[newIndex];
                  geom.positions[newIndex]            = oldPositions[oldIndex];
                  geom.clusterLocalVertices[newIndex] = newIndex;
                }
              }
            },
            numThreads);
      }

      // rebuild triangle buffer accounting for cluster order
      {

        uint32_t triOffset = 0;
        for(size_t c = 0; c < geom.numClusters; c++)
        {
          shaderio::Cluster& cluster = geom.clusters[c];

          cluster.firstTriangle = triOffset;
          triOffset += cluster.numTriangles;
          m_clusterTriangleHistogram[cluster.numTriangles]++;
          m_clusterVertexHistogram[cluster.numVertices]++;
        }

        geom.triangles.resize(triOffset);
        geom.numTriangles = triOffset;

        glm::uvec3*     triangles             = geom.triangles.data();
        const uint32_t* clusterLocalVertices  = geom.clusterLocalVertices.data();
        const uint8_t*  clusterLocalTriangles = geom.clusterLocalTriangles.data();

        nvh::parallel_ranges(
            geom.numClusters,
            [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadIdx) {
              for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
              {
                shaderio::Cluster& cluster = geom.clusters[idx];

                for(uint32_t t = 0; t < cluster.numTriangles; t++)
                {
                  glm::uvec3 localVertices = {clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 0],
                                              clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 1],
                                              clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 2]};

                  assert(localVertices.x < cluster.numVertices);
                  assert(localVertices.y < cluster.numVertices);
                  assert(localVertices.z < cluster.numVertices);

                  glm::uvec3 globalVertices = {clusterLocalVertices[localVertices.x + cluster.firstLocalVertex],
                                               clusterLocalVertices[localVertices.y + cluster.firstLocalVertex],
                                               clusterLocalVertices[localVertices.z + cluster.firstLocalVertex]};

                  triangles[cluster.firstTriangle + t] = globalVertices;
                }
              }
            },
            numThreads);

        shaderio::Cluster& cluster = geom.clusters[geom.numClusters - 1];
        geom.clusterLocalTriangles.resize(cluster.firstLocalTriangle + cluster.numTriangles * 3);
        geom.clusterLocalVertices.resize(cluster.firstLocalVertex + cluster.numVertices);
        geom.clusterLocalTriangles.shrink_to_fit();
        geom.clusterLocalVertices.shrink_to_fit();
      }

      // build bboxes
      {
        geom.clusterBboxes.resize(geom.numClusters);

        const glm::vec3* positions            = geom.positions.data();
        const uint32_t*  clusterLocalVertices = geom.clusterLocalVertices.data();

        nvh::parallel_ranges(
            geom.numClusters,
            [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadIdx) {
              for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
              {
                shaderio::Cluster& cluster = geom.clusters[idx];

                shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
                for(uint32_t v = 0; v < cluster.numVertices; v++)
                {
                  uint32_t  vertexIndex = clusterLocalVertices[cluster.firstLocalVertex + v];
                  glm::vec3 pos         = positions[vertexIndex];

                  bbox.lo = glm::min(bbox.lo, pos);
                  bbox.hi = glm::max(bbox.hi, pos);
                }

                geom.clusterBboxes[idx] = bbox;
              }
            },
            numThreads);
      }
    }


    m_numClusters += geom.numClusters;

    m_maxPerGeometryClusters = std::max(m_maxPerGeometryClusters, geom.numClusters);
    m_maxPerGeometryClusterVertices = std::max(m_maxPerGeometryClusterVertices, uint32_t(geom.clusterLocalVertices.size()));
  }

  nvclusterDestroyContext(nvclusterContext);

  double endTime = clock.getMicroSeconds();
  LOGI("Scene cluster build time: %f milliseconds\n", (endTime - startTime) / 1000.0f);

  // shrink, to adjust for actual max cluster vertices
  m_clusterVertexHistogram.resize(m_config.clusterVertices + 1);

  m_clusterTriangleHistogramMax = 0u;
  m_clusterVertexHistogramMax   = 0u;
  for(size_t i = 0; i < m_clusterTriangleHistogram.size(); i++)
  {
    m_clusterTriangleHistogramMax = std::max(m_clusterTriangleHistogramMax, m_clusterTriangleHistogram[i]);
  }
  for(size_t i = 0; i < m_clusterVertexHistogram.size(); i++)
  {
    m_clusterVertexHistogramMax = std::max(m_clusterVertexHistogramMax, m_clusterVertexHistogram[i]);
  }

  if(m_config.clusterStripify && (numTotalStrips > 0))
  {
    LOGI("Average triangles per strip %.2f\n", double(numTotalTriangles) / double(numTotalStrips));
  }

  return true;
}

}  // namespace animatedclusters
