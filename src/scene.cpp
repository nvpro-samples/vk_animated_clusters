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

#include <cassert>

#include <meshoptimizer.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <nvutils/parallel_work.hpp>

#include "scene.hpp"

namespace animatedclusters {

void Scene::ProcessingInfo::init(float processingThreadsPct)
{
  numPoolThreadsOriginal = nvutils::get_thread_pool().get_thread_count();

  numPoolThreads = numPoolThreadsOriginal;
  if(processingThreadsPct > 0.0f && processingThreadsPct < 1.0f)
  {
    numPoolThreads = std::min(numPoolThreadsOriginal,
                              std::max(1u, uint32_t(ceilf(float(numPoolThreadsOriginal) * processingThreadsPct))));

    if(numPoolThreads != numPoolThreadsOriginal)
      nvutils::get_thread_pool().reset(numPoolThreads);
  }
}

void Scene::ProcessingInfo::setupParallelism(size_t geometryCount_)
{
  geometryCount = geometryCount_;

  bool preferInnerParallelism = geometryCount < numPoolThreads;

  numOuterThreads = preferInnerParallelism ? 1 : numPoolThreads;
  numInnerThreads = preferInnerParallelism ? numPoolThreads : 1;

  nvcluster_ContextCreateInfo clusterContextInfo;
  clusterContextInfo.parallelize = preferInnerParallelism ? 1 : 0;
  nvclusterCreateContext(&clusterContextInfo, &clusterContext);
}

void Scene::ProcessingInfo::logBegin()
{
  LOGI("... geometry load & processing: geometries %llu, threads outer %d inner %d\n", geometryCount, numOuterThreads, numInnerThreads);

  startTime = clock.getMicroseconds();
}

void Scene::ProcessingInfo::logCompletedGeometry()
{
  std::lock_guard lock(progressMutex);

  progressGeometriesCompleted++;

  // statistics
  const uint32_t precentageGranularity = 5;
  uint32_t       percentage            = uint32_t(size_t(progressGeometriesCompleted * 100) / geometryCount);
  percentage                           = (percentage / precentageGranularity) * precentageGranularity;

  if(percentage > progressLastPercentage)
  {
    progressLastPercentage = percentage;
    LOGI("... geometry load & processing: %3d%%\n", percentage);
  }
}

void Scene::ProcessingInfo::logEnd()
{
  double endTime = clock.getMicroseconds();

  LOGI("... geometry load & processing: %f milliseconds\n", (endTime - startTime) / 1000.0f);
}

void Scene::ProcessingInfo::deinit()
{
  if(clusterContext)
    nvclusterDestroyContext(clusterContext);

  if(numPoolThreads != numPoolThreadsOriginal)
    nvutils::get_thread_pool().reset(numPoolThreadsOriginal);
}
bool Scene::init(const std::filesystem::path& filePath, const SceneConfig& config, Resources& res)
{
  m_config = config;

  m_clusterTriangleHistogram.resize(m_config.clusterTriangles + 1, 0);
  m_clusterVertexHistogram.resize(m_config.clusterVertices + 1, 0);

  ProcessingInfo processingInfo;
  processingInfo.init(config.processingThreadsPct);

  if(!loadGLTF(processingInfo, filePath))
  {
    return false;
  }

  processingInfo.deinit();

  m_clusterTriangleHistogramMax = 0u;
  m_clusterVertexHistogramMax   = 0u;
  for(size_t i = 0; i < m_clusterTriangleHistogram.size(); i++)
  {
    m_clusterTriangleHistogramMax = std::max(m_clusterTriangleHistogramMax, m_clusterTriangleHistogram[i]);
    if(m_clusterTriangleHistogram[i])
      m_maxClusterTriangles = uint32_t(i);
  }
  for(size_t i = 0; i < m_clusterVertexHistogram.size(); i++)
  {
    m_clusterVertexHistogramMax = std::max(m_clusterVertexHistogramMax, m_clusterVertexHistogram[i]);
    if(m_clusterVertexHistogram[i])
      m_maxClusterVertices = uint32_t(i);
  }

  if(m_config.clusterStripify && (processingInfo.numTotalStrips > 0))
  {
    LOGI("Average triangles per strip %.2f\n", double(processingInfo.numTotalTriangles) / double(processingInfo.numTotalStrips));
  }

  computeInstanceBBoxes();

  for(auto& geometry : m_geometries)
  {
    m_maxPerGeometryTriangles       = std::max(m_maxPerGeometryTriangles, geometry.numTriangles);
    m_maxPerGeometryVertices        = std::max(m_maxPerGeometryVertices, geometry.numVertices);
    m_maxPerGeometryClusters        = std::max(m_maxPerGeometryClusters, geometry.numClusters);
    m_maxPerGeometryClusterVertices = std::max(m_maxPerGeometryClusterVertices, geometry.numClusterVertices);
    m_numTriangles += geometry.numTriangles;
    m_numClusters += geometry.numClusters;
  }

  initGpuBuffers(res);

  return true;
}

void Scene::deinit(Resources& res)
{
  for(auto& geometry : m_geometries)
  {
    res.m_allocator.destroyBuffer(geometry.positionsBuffer);
    res.m_allocator.destroyBuffer(geometry.trianglesBuffer);
    res.m_allocator.destroyBuffer(geometry.clustersBuffer);
    res.m_allocator.destroyBuffer(geometry.clusterLocalTrianglesBuffer);
    res.m_allocator.destroyBuffer(geometry.clusterLocalVerticesBuffer);
    res.m_allocator.destroyBuffer(geometry.clusterBboxesBuffer);
  }
}

void Scene::computeInstanceBBoxes()
{
  m_bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};

  for(auto& instance : m_instances)
  {
    assert(instance.geometryID <= m_geometries.size());

    const Geometry& geometry = m_geometries[instance.geometryID];

    instance.bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};

    for(uint32_t v = 0; v < 8; v++)
    {
      bool x = (v & 1) != 0;
      bool y = (v & 2) != 0;
      bool z = (v & 4) != 0;

      glm::bvec3 weight(x, y, z);
      glm::vec3  corner = glm::mix(geometry.bbox.lo, geometry.bbox.hi, weight);
      corner            = instance.matrix * glm::vec4(corner, 1.0f);
      instance.bbox.lo  = glm::min(instance.bbox.lo, corner);
      instance.bbox.hi  = glm::max(instance.bbox.hi, corner);
    }

    m_bbox.lo = glm::min(m_bbox.lo, instance.bbox.lo);
    m_bbox.hi = glm::max(m_bbox.hi, instance.bbox.hi);
  }
}

void Scene::initGpuBuffers(Resources& res)
{
  m_sceneClusterMemBytes  = 0;
  m_sceneTriangleMemBytes = 0;

  Resources::BatchedUploader uploader(res);

  for(auto& geometry : m_geometries)
  {
    if(geometry.positions.size())
    {
      res.m_allocator.createBuffer(geometry.positionsBuffer, sizeof(glm::vec3) * geometry.positions.size(),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NVVK_DBG_NAME(geometry.positionsBuffer.buffer);

      uploader.uploadBuffer(geometry.positionsBuffer, geometry.positions.data());

      m_sceneClusterMemBytes += geometry.positionsBuffer.bufferSize;
      m_sceneTriangleMemBytes += geometry.positionsBuffer.bufferSize;
    }
    if(geometry.triangles.size())
    {
      res.m_allocator.createBuffer(geometry.trianglesBuffer, sizeof(glm::uvec3) * geometry.triangles.size(),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NVVK_DBG_NAME(geometry.trianglesBuffer.buffer);

      uploader.uploadBuffer(geometry.trianglesBuffer, geometry.triangles.data());

      // animation still needs triangles for normals, even if clusters are used
      m_sceneClusterMemBytes += geometry.trianglesBuffer.bufferSize;
      m_sceneTriangleMemBytes += geometry.trianglesBuffer.bufferSize;
    }
    if(geometry.clusters.size())
    {
      res.m_allocator.createBuffer(geometry.clustersBuffer, sizeof(shaderio::Cluster) * geometry.clusters.size(),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NVVK_DBG_NAME(geometry.clustersBuffer.buffer);

      uploader.uploadBuffer(geometry.clustersBuffer, geometry.clusters.data());

      m_sceneClusterMemBytes += geometry.clustersBuffer.bufferSize;
    }
    if(geometry.clusterLocalTriangles.size())
    {
      res.m_allocator.createBuffer(
          geometry.clusterLocalTrianglesBuffer, sizeof(uint8_t) * geometry.clusterLocalTriangles.size(),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NVVK_DBG_NAME(geometry.clusterLocalTrianglesBuffer.buffer);

      uploader.uploadBuffer(geometry.clusterLocalTrianglesBuffer, geometry.clusterLocalTriangles.data());

      m_sceneClusterMemBytes += geometry.clusterLocalTrianglesBuffer.bufferSize;
    }
    if(geometry.clusterLocalVertices.size() && !m_config.clusterDedicatedVertices)
    {
      res.m_allocator.createBuffer(geometry.clusterLocalVerticesBuffer, sizeof(uint32_t) * geometry.clusterLocalVertices.size(),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NVVK_DBG_NAME(geometry.clusterLocalVerticesBuffer.buffer);

      uploader.uploadBuffer(geometry.clusterLocalVerticesBuffer, geometry.clusterLocalVertices.data());

      m_sceneClusterMemBytes += geometry.clusterLocalVerticesBuffer.bufferSize;
    }
    if(geometry.clusterBboxes.size())
    {
      res.m_allocator.createBuffer(geometry.clusterBboxesBuffer, sizeof(shaderio::BBox) * geometry.clusterBboxes.size(),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
      NVVK_DBG_NAME(geometry.clusterBboxesBuffer.buffer);

      uploader.uploadBuffer(geometry.clusterBboxesBuffer, geometry.clusterBboxes.data());

      m_sceneClusterMemBytes += geometry.clusterBboxesBuffer.bufferSize;
    }
  }

  uploader.flush();
}

void Scene::processGeometry(ProcessingInfo& processingInfo, Geometry& geometry)
{
  if(!geometry.numTriangles)
    return;

  buildGeometryClusters(processingInfo, geometry);

  if(!geometry.numClusters)
    return;

  if(m_config.clusterStripify)
  {
    buildGeometryClusterStrips(processingInfo, geometry);
  }

  buildGeometryClusterBboxes(processingInfo, geometry);

  if(m_config.clusterDedicatedVertices)
  {
    // give each cluster its own set of vertices, so require only
    // the local cluster 8-bit triangle indices
    buildGeometryClusterVertices(processingInfo, geometry);

    // no longer need vertex indirection
    geometry.clusterLocalVertices = std::vector<uint32_t>();
  }

  rebuildGeometryTriangles(processingInfo, geometry);
}

void Scene::buildGeometryClusters(ProcessingInfo& processingInfo, Geometry& geometry)
{
  uint32_t numInnerThreads = processingInfo.numInnerThreads;

  if(m_config.clusterNvLibrary)
  {
    std::vector<nvcluster_AABB> triangleAABBs(geometry.numTriangles);
    std::vector<glm::vec3>      triangleCentroids(geometry.numTriangles);

    static_assert(std::atomic_uint32_t::is_always_lock_free && sizeof(uint32_t) == sizeof(std::atomic_uint32_t));

    nvutils::parallel_batches_pooled(
        geometry.numTriangles,
        [&](uint64_t t, uint32_t threadInnerIdx) {
          glm::uvec3 triangleIndices = geometry.triangles[t];

          glm::vec3 vertexA = geometry.positions[triangleIndices.x];
          glm::vec3 vertexB = geometry.positions[triangleIndices.y];
          glm::vec3 vertexC = geometry.positions[triangleIndices.z];

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
        },
        numInnerThreads);

    nvcluster_Input input;
    input.itemCount         = geometry.numTriangles;
    input.itemBoundingBoxes = triangleAABBs.data();
    input.itemCentroids     = reinterpret_cast<const nvcluster_Vec3f*>(triangleCentroids.data());

    nvcluster_Config config = m_config.clusterNvConfig;
    config.maxClusterSize   = m_config.clusterTriangles;

    bool useVertexLimit = m_config.clusterVertices < m_config.clusterTriangles * 3;
    if(useVertexLimit)
    {
      config.itemVertexCount    = 3;
      config.maxClusterVertices = m_config.clusterVertices;

      input.itemVertices = reinterpret_cast<const uint32_t*>(geometry.triangles.data());
      input.vertexCount  = geometry.numVertices;
    }

    nvcluster::ClusterStorage storage;
    nvcluster_Result nvclusterResult = nvcluster::generateClusters(processingInfo.clusterContext, config, input, storage);
    assert(nvclusterResult == NVCLUSTER_SUCCESS);

    size_t numClusters   = storage.clusterItemRanges.size();
    geometry.numClusters = uint32_t(numClusters);

    if(numClusters)
    {
      geometry.clusterLocalTriangles.resize(geometry.numTriangles * 3);
      geometry.clusterLocalVertices.resize(geometry.numTriangles * 3);
      geometry.clusters.resize(numClusters);

      // linearize triangle offsets
      uint32_t firstLocalTriangleOffset = 0;
      for(size_t c = 0; c < numClusters; c++)
      {
        shaderio::Cluster& cluster = geometry.clusters[c];
        cluster.numTriangles       = storage.clusterItemRanges[c].count;
        cluster.firstLocalTriangle = firstLocalTriangleOffset;
        firstLocalTriangleOffset += cluster.numTriangles * 3;

        assert(cluster.numVertices <= m_config.clusterTriangles);
      }

      std::vector<uint32_t> threadCacheEarly(numInnerThreads * 256 * 2);

      // fill clusters
      nvutils::parallel_batches_pooled(
          numClusters,
          [&](uint64_t idx, uint32_t threadInnerIdx) {
            shaderio::Cluster& cluster      = geometry.clusters[idx];
            nvcluster_Range    clusterRange = storage.clusterItemRanges[idx];

            uint8_t*  localTriangles        = &geometry.clusterLocalTriangles[cluster.firstLocalTriangle];
            uint32_t* localVertices         = &geometry.clusterLocalVertices[cluster.firstLocalTriangle];
            uint32_t* localItems            = &storage.items[clusterRange.offset];
            uint32_t* vertexCacheEarlyValue = &threadCacheEarly[threadInnerIdx * 256 * 2];
            uint32_t* vertexCacheEarlyPos   = vertexCacheEarlyValue + 256;
            memset(vertexCacheEarlyValue, ~0, sizeof(uint32_t) * 256);

            uint32_t numVertices = 0;
            uint32_t numIndices  = 0;

            for(uint32_t t = 0; t < cluster.numTriangles; t++)
            {
              glm::uvec3 triangleIndices = geometry.triangles[localItems[t]];
              for(uint32_t k = 0; k < 3; k++)
              {
                uint32_t vertexIndex = triangleIndices[k];
                bool     found       = false;

                // quick early out
                if(vertexCacheEarlyValue[vertexIndex & 0xFF] == vertexIndex)
                {
                  localTriangles[numIndices++] = uint8_t(vertexCacheEarlyPos[vertexIndex & 0xFF]);
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
                  vertexCacheEarlyValue[vertexIndex & 0xFF] = vertexIndex;
                  vertexCacheEarlyPos[vertexIndex & 0xFF]   = numVertices;

                  localTriangles[numIndices++] = numVertices;
                  localVertices[numVertices++] = vertexIndex;
                }
              }
            }
            assert(numIndices == cluster.numTriangles * 3);

            cluster.numVertices = numVertices;

            assert(cluster.numVertices <= m_config.clusterVertices);

            // update stats
            reinterpret_cast<std::atomic_uint32_t*>(m_clusterTriangleHistogram.data())[cluster.numTriangles]++;
            reinterpret_cast<std::atomic_uint32_t*>(m_clusterVertexHistogram.data())[cluster.numVertices]++;
          },
          numInnerThreads);

      // compact local vertices
      uint32_t writeOffset = 0;
      for(size_t c = 0; c < numClusters; c++)
      {
        shaderio::Cluster& cluster = geometry.clusters[c];
        cluster.firstLocalVertex   = writeOffset;

        memmove(&geometry.clusterLocalVertices[writeOffset], &geometry.clusterLocalVertices[cluster.firstLocalTriangle],
                sizeof(uint32_t) * cluster.numVertices);

        writeOffset += uint32_t(cluster.numVertices);
      }
      geometry.clusterLocalVertices.resize(writeOffset);
      geometry.clusterLocalVertices.shrink_to_fit();
    }
  }
  else
  {
    // we allow smaller clusters to be generated when that significantly improves their bounds
    size_t minTriangles = (m_config.clusterTriangles / 4) & ~3;

    std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(geometry.numTriangles * 3, m_config.clusterVertices, minTriangles));
    geometry.clusterLocalTriangles.resize(meshlets.size() * m_config.clusterTriangles * 3);
    geometry.clusterLocalVertices.resize(meshlets.size() * m_config.clusterVertices);

    size_t numClusters =
        m_config.clusterMeshoptSpatialFill > 1.01f ?
            meshopt_buildMeshletsFlex(meshlets.data(), geometry.clusterLocalVertices.data(),
                                      geometry.clusterLocalTriangles.data(), (uint32_t*)geometry.triangles.data(),
                                      geometry.triangles.size() * 3, (float*)geometry.positions.data(),
                                      geometry.numVertices, sizeof(glm::vec3), m_config.clusterVertices,
                                      minTriangles, m_config.clusterTriangles, 0.0f, 2.0f) :
            meshopt_buildMeshletsSpatial(meshlets.data(), geometry.clusterLocalVertices.data(),
                                         geometry.clusterLocalTriangles.data(), (uint32_t*)geometry.triangles.data(),
                                         geometry.triangles.size() * 3, (float*)geometry.positions.data(),
                                         geometry.numVertices, sizeof(glm::vec3), m_config.clusterVertices,
                                         minTriangles, m_config.clusterTriangles, m_config.clusterMeshoptSpatialFill);

    geometry.numClusters = uint32_t(numClusters);

    if(geometry.numClusters)
    {
      geometry.clusters.resize(geometry.numClusters);
      geometry.clusters.shrink_to_fit();

      for(size_t c = 0; c < numClusters; c++)
      {
        meshopt_Meshlet&   meshlet = meshlets[c];
        shaderio::Cluster& cluster = geometry.clusters[c];

        cluster.numTriangles       = meshlet.triangle_count;
        cluster.numVertices        = meshlet.vertex_count;
        cluster.firstLocalTriangle = meshlet.triangle_offset;
        cluster.firstLocalVertex   = meshlet.vertex_offset;

        // update stats
        reinterpret_cast<std::atomic_uint32_t*>(m_clusterTriangleHistogram.data())[cluster.numTriangles]++;
        reinterpret_cast<std::atomic_uint32_t*>(m_clusterVertexHistogram.data())[cluster.numVertices]++;
      }
    }
  }

  if(geometry.numClusters)
  {
    shaderio::Cluster& cluster = geometry.clusters[geometry.numClusters - 1];
    geometry.clusterLocalTriangles.resize(cluster.firstLocalTriangle + cluster.numTriangles * 3);
    geometry.clusterLocalVertices.resize(cluster.firstLocalVertex + cluster.numVertices);
    geometry.clusterLocalTriangles.shrink_to_fit();
    geometry.clusterLocalVertices.shrink_to_fit();

    geometry.numClusterVertices = uint32_t(geometry.clusterLocalVertices.size());
  }
}


void Scene::buildGeometryClusterStrips(ProcessingInfo& processingInfo, Geometry& geometry)
{
  uint32_t numInnerThreads = processingInfo.numInnerThreads;

  uint32_t numMaxTriangles  = m_config.clusterTriangles;
  uint32_t perThreadIndices = numMaxTriangles * 3 * 2 + uint32_t(meshopt_stripifyBound(numMaxTriangles * 3));

  std::atomic_uint32_t  numStrips = 0;
  std::vector<uint32_t> threadIndices(numInnerThreads * perThreadIndices);

  nvutils::parallel_ranges_pooled(
      geometry.numClusters,
      [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadInnerIdx) {
        for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
        {
          shaderio::Cluster& cluster = geometry.clusters[idx];

          uint32_t* meshletIndices      = &threadIndices[threadInnerIdx * perThreadIndices];
          uint32_t* meshletOptim        = meshletIndices + cluster.numTriangles * 3;
          uint32_t* meshletStripIndices = meshletOptim + cluster.numTriangles * 3;

          // convert u8 to u32
          for(uint32_t i = 0; i < uint32_t(cluster.numTriangles) * 3; i++)
          {
            meshletIndices[i] = geometry.clusterLocalTriangles[cluster.firstLocalTriangle + i];
          }

          meshopt_optimizeVertexCache(meshletOptim, meshletIndices, cluster.numTriangles * 3, cluster.numVertices);
          size_t stripIndexCount =
              meshopt_stripify(meshletStripIndices, meshletOptim, cluster.numTriangles * 3, cluster.numVertices, ~0);
          size_t newIndexCount = meshopt_unstripify(meshletIndices, meshletStripIndices, stripIndexCount, ~0);

          cluster.numTriangles = uint32_t(newIndexCount / 3);

          for(uint32_t i = 0; i < uint32_t(newIndexCount); i++)
          {
            geometry.clusterLocalTriangles[cluster.firstLocalTriangle + i] = uint8_t(meshletIndices[i]);
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
      numInnerThreads);

  processingInfo.numTotalTriangles += geometry.numTriangles;
  processingInfo.numTotalStrips += numStrips;
}

void Scene::buildGeometryClusterBboxes(ProcessingInfo& processingInfo, Geometry& geometry)
{
  geometry.clusterBboxes.resize(geometry.numClusters);

  const glm::vec3* positions             = geometry.positions.data();
  const uint32_t*  clusterLocalVertices  = geometry.clusterLocalVertices.data();
  const uint8_t*   clusterLocalTriangles = geometry.clusterLocalTriangles.data();

  nvutils::parallel_ranges_pooled(
      geometry.numClusters,
      [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadInnerIdx) {
        for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
        {
          shaderio::Cluster& cluster = geometry.clusters[idx];

          shaderio::BBox bbox = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
          for(uint32_t v = 0; v < cluster.numVertices; v++)
          {
            uint32_t  vertexIndex = clusterLocalVertices[cluster.firstLocalVertex + v];
            glm::vec3 pos         = positions[vertexIndex];

            bbox.lo = glm::min(bbox.lo, pos);
            bbox.hi = glm::max(bbox.hi, pos);
          }

          geometry.clusterBboxes[idx] = bbox;
        }
      },
      processingInfo.numInnerThreads);
}

void Scene::buildGeometryClusterVertices(ProcessingInfo& processingInfo, Geometry& geometry)
{
  // build per-cluster vertices

  std::vector<glm::vec3> oldPositionsData = std::move(geometry.positions);

  geometry.positions.resize(geometry.numClusterVertices);
  geometry.numVertices = uint32_t(geometry.positions.size());

  const glm::vec3* oldPositions         = oldPositionsData.data();
  glm::vec3*       newPositions         = geometry.positions.data();
  uint32_t*        clusterLocalVertices = geometry.clusterLocalVertices.data();

  for(uint32_t c = 0; c < geometry.numClusters; c++)
  {
    shaderio::Cluster& cluster = geometry.clusters[c];

    for(uint32_t v = 0; v < cluster.numVertices; v++)
    {
      uint32_t oldIdx                                    = clusterLocalVertices[v + cluster.firstLocalVertex];
      clusterLocalVertices[v + cluster.firstLocalVertex] = v + cluster.firstLocalVertex;
      newPositions[v + cluster.firstLocalVertex]         = oldPositions[oldIdx];
    }
  }
}

void Scene::rebuildGeometryTriangles(ProcessingInfo& processingInfo, Geometry& geometry)
{
  // rebuild triangle buffer accounting for cluster order
  // in the rare event that cluster building filtered out original triangles

  uint32_t triOffset = 0;
  for(size_t c = 0; c < geometry.numClusters; c++)
  {
    shaderio::Cluster& cluster = geometry.clusters[c];

    cluster.firstTriangle = triOffset;
    triOffset += cluster.numTriangles;
  }

  geometry.triangles.resize(triOffset);
  geometry.numTriangles = triOffset;

  glm::uvec3*     triangles             = geometry.triangles.data();
  const uint32_t* clusterLocalVertices  = geometry.clusterLocalVertices.data();
  const uint8_t*  clusterLocalTriangles = geometry.clusterLocalTriangles.data();

  nvutils::parallel_ranges_pooled(
      geometry.numClusters,
      [&](uint64_t idxBegin, uint64_t idxEnd, uint32_t threadInnerIdx) {
        for(uint64_t idx = idxBegin; idx < idxEnd; idx++)
        {
          shaderio::Cluster& cluster = geometry.clusters[idx];

          for(uint32_t t = 0; t < cluster.numTriangles; t++)
          {
            glm::uvec3 localVertices = {clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 0],
                                        clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 1],
                                        clusterLocalTriangles[cluster.firstLocalTriangle + t * 3 + 2]};

            assert(localVertices.x < cluster.numVertices);
            assert(localVertices.y < cluster.numVertices);
            assert(localVertices.z < cluster.numVertices);

            glm::uvec3 globalVertices = {localVertices.x + cluster.firstLocalVertex, localVertices.y + cluster.firstLocalVertex,
                                         localVertices.z + cluster.firstLocalVertex};

            if(!m_config.clusterDedicatedVertices)
            {
              // need one more indirection
              globalVertices = {clusterLocalVertices[globalVertices.x], clusterLocalVertices[globalVertices.y],
                                clusterLocalVertices[globalVertices.z]};
            }

            triangles[cluster.firstTriangle + t] = globalVertices;
          }
        }
      },
      processingInfo.numInnerThreads);
}

}  // namespace animatedclusters
