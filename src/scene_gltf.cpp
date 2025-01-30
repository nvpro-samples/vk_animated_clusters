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

#include <float.h>

#include <glm/gtc/type_ptr.hpp>
#include <cgltf.h>
#include <nvh/filemapping.hpp>

#include "scene.hpp"

namespace {
struct FileMappingList
{
  struct Entry
  {
    nvh::FileReadMapping mapping;
    int64_t              refCount = 1;
  };
  std::unordered_map<std::string, Entry>       m_nameToMapping;
  std::unordered_map<const void*, std::string> m_dataToName;
#ifdef _DEBUG
  int64_t m_openBias = 0;
#endif

  bool open(const char* path, size_t* size, void** data)
  {
#ifdef _DEBUG
    m_openBias++;
#endif

    std::string pathStr(path);

    auto it = m_nameToMapping.find(pathStr);
    if(it != m_nameToMapping.end())
    {
      *data = const_cast<void*>(it->second.mapping.data());
      *size = it->second.mapping.size();
      it->second.refCount++;
      return true;
    }

    Entry entry;
    if(entry.mapping.open(path))
    {
      const void* mappingData = entry.mapping.data();
      *data                   = const_cast<void*>(mappingData);
      *size                   = entry.mapping.size();
      m_dataToName.insert({mappingData, pathStr});
      m_nameToMapping.insert({pathStr, std::move(entry)});
      return true;
    }

    return false;
  }

  void close(void* data)
  {
#ifdef _DEBUG
    m_openBias--;
#endif
    auto itName = m_dataToName.find(data);
    if(itName != m_dataToName.end())
    {
      auto itMapping = m_nameToMapping.find(itName->second);
      if(itMapping != m_nameToMapping.end())
      {
        itMapping->second.refCount--;

        if(!itMapping->second.refCount)
        {
          m_nameToMapping.erase(itMapping);
          m_dataToName.erase(itName);
        }
      }
    }
  }

  ~FileMappingList()
  {
#ifdef _DEBUG
    assert(m_openBias == 0 && "open/close bias wrong");
#endif
    assert(m_nameToMapping.empty() && m_dataToName.empty() && "not all opened files were closed");
  }
};

const uint8_t* cgltf_buffer_view_data(const cgltf_buffer_view* view)
{
  if(view->data)
    return (const uint8_t*)view->data;

  if(!view->buffer->data)
    return NULL;

  const uint8_t* result = (const uint8_t*)view->buffer->data;
  result += view->offset;
  return result;
}

cgltf_result cgltf_read(const struct cgltf_memory_options* memory_options,
                        const struct cgltf_file_options*   file_options,
                        const char*                        path,
                        cgltf_size*                        size,
                        void**                             data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  if(mappings->open(path, size, data))
  {
    return cgltf_result_success;
  }

  return cgltf_result_io_error;
}

void cgltf_release(const struct cgltf_memory_options* memory_options, const struct cgltf_file_options* file_options, void* data)
{
  FileMappingList* mappings = (FileMappingList*)file_options->user_data;
  mappings->close(data);
}

// Defines a unique_ptr that can be used for cgltf_data objects.
// Freeing a unique_cgltf_ptr calls cgltf_free, instead of delete.
// This can be constructed using unique_cgltf_ptr foo(..., &cgltf_free).
using unique_cgltf_ptr = std::unique_ptr<cgltf_data, decltype(&cgltf_free)>;


// Traverses the glTF node and any of its children, adding a MeshInstance to
// the meshSet for each referenced glTF primitive.
void addInstancesFromNode(std::vector<animatedclusters::Scene::Instance>& instances,
                          const cgltf_data*                               data,
                          const cgltf_node*                               node,
                          const glm::mat4                                 parentObjToWorldTransform = glm::mat4(1))
{
  if(node == nullptr)
    return;

  // Compute this node's object-to-world transform.
  // See https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_004_ScenesNodes.md .
  // Note that this depends on glm::mat4 being column-major.
  // The documentation above also means that vectors are multiplied on the right.
  glm::mat4 localNodeTransform(1);
  cgltf_node_transform_local(node, glm::value_ptr(localNodeTransform));
  const glm::mat4 nodeObjToWorldTransform = parentObjToWorldTransform * localNodeTransform;

  // If this node has a mesh, add instances for its primitives.
  if(node->mesh != nullptr)
  {
    const ptrdiff_t meshIndex = (node->mesh) - data->meshes;

    animatedclusters::Scene::Instance instance{};
    instance.geometryID = uint32_t(meshIndex);
    instance.matrix     = nodeObjToWorldTransform;

    instances.push_back(instance);
  }

  // Recurse over any children of this node.
  const size_t numChildren = node->children_count;
  for(size_t childIdx = 0; childIdx < numChildren; childIdx++)
  {
    addInstancesFromNode(instances, data, node->children[childIdx], nodeObjToWorldTransform);
  }
}

}  // namespace


namespace animatedclusters {
bool Scene::loadGLTF(const char* filename)
{
  // Parse the glTF file using cgltf
  cgltf_options options = {};

  FileMappingList mappings;
  options.file.read      = cgltf_read;
  options.file.release   = cgltf_release;
  options.file.user_data = &mappings;

  cgltf_result     cgltfResult;
  unique_cgltf_ptr data = unique_cgltf_ptr(nullptr, &cgltf_free);
  {
    // We have this local pointer followed by an ownership transfer here
    // because cgltf_parse_file takes a pointer to a pointer to cgltf_data.
    cgltf_data* rawData = nullptr;
    cgltfResult         = cgltf_parse_file(&options, filename, &rawData);
    data                = unique_cgltf_ptr(rawData, &cgltf_free);
  }
  // Check for errors; special message for legacy files
  if(cgltfResult == cgltf_result_legacy_gltf)
  {
    LOGE(
        "loadGLTF: This glTF file is an unsupported legacy file - probably glTF 1.0, while cgltf only supports glTF "
        "2.0 files. Please load a glTF 2.0 file instead.\n");
    return false;
  }
  else if((cgltfResult != cgltf_result_success) || (data == nullptr))
  {
    LOGE("loadGLTF: cgltf_parse_file failed. Is this a valid glTF file? (cgltf result: %d)\n", cgltfResult);
    return false;
  }

  // Perform additional validation.
  cgltfResult = cgltf_validate(data.get());
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file could be parsed, but cgltf_validate failed. Consider using the glTF Validator at "
        "https://github.khronos.org/glTF-Validator/ to see if the non-displacement parts of the glTF file are correct. "
        "(cgltf result: %d)\n",
        cgltfResult);
    return false;
  }

  // For now, also tell cgltf to go ahead and load all buffers.
  cgltfResult = cgltf_load_buffers(&options, data.get(), filename);
  if(cgltfResult != cgltf_result_success)
  {
    LOGE(
        "loadGLTF: The glTF file was valid, but cgltf_load_buffers failed. Are the glTF file's referenced file paths "
        "valid? (cgltf result: %d)\n",
        cgltfResult);
    return false;
  }

  m_geometries.resize(data->meshes_count);

  for(size_t meshIdx = 0; meshIdx < data->meshes_count; meshIdx++)
  {
    const cgltf_mesh gltfMesh = data->meshes[meshIdx];
    Geometry&        geom     = m_geometries[meshIdx];
    geom.bbox                 = {{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};


    // count pass
    geom.numTriangles = 0;
    geom.numVertices  = 0;
    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        continue;
      }

      // If the mesh has no attributes, there's nothing we can do
      if(gltfPrim->attributes_count == 0)
      {
        continue;
      }

      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor*  accessor   = gltfAttrib.data;

        // TODO: Can we assume alignment in order to make these a single read_float call?
        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          geom.numVertices += (uint32_t)accessor->count;
          break;
        }
      }

      geom.numTriangles += (uint32_t)gltfPrim->indices->count / 3;
    }

    geom.positions.resize(geom.numVertices);
    geom.triangles.resize(geom.numTriangles);

    // fill pass

    uint32_t offsetVertices  = 0;
    uint32_t offsetTriangles = 0;

    for(size_t primIdx = 0; primIdx < gltfMesh.primitives_count; primIdx++)
    {
      cgltf_primitive* gltfPrim = &gltfMesh.primitives[primIdx];

      if(gltfPrim->type != cgltf_primitive_type_triangles)
      {
        continue;
      }

      // If the mesh has no attributes, there's nothing we can do
      if(gltfPrim->attributes_count == 0)
      {
        continue;
      }

      for(size_t attribIdx = 0; attribIdx < gltfPrim->attributes_count; attribIdx++)
      {
        const cgltf_attribute& gltfAttrib = gltfPrim->attributes[attribIdx];
        const cgltf_accessor*  accessor   = gltfAttrib.data;

        // TODO: Can we assume alignment in order to make these a single read_float call?
        if(strcmp(gltfAttrib.name, "POSITION") == 0)
        {
          glm::vec3* writePositions = geom.positions.data() + offsetVertices;

          if(accessor->component_type == cgltf_component_type_r_32f && accessor->type == cgltf_type_vec3
             && accessor->stride == sizeof(glm::vec3))
          {
            const glm::vec3* readPositions = (const glm::vec3*)(cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset);
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec3 tmp     = readPositions[i];
              writePositions[i] = tmp;
              geom.bbox.lo      = glm::min(geom.bbox.lo, tmp);
              geom.bbox.hi      = glm::max(geom.bbox.hi, tmp);
            }
          }
          else
          {
            for(size_t i = 0; i < accessor->count; i++)
            {
              glm::vec3 tmp;
              cgltf_accessor_read_float(accessor, i, &tmp.x, 3);
              writePositions[i] = tmp;
              geom.bbox.lo      = glm::min(geom.bbox.lo, tmp);
              geom.bbox.hi      = glm::max(geom.bbox.hi, tmp);
            }
          }

          offsetVertices += (uint32_t)accessor->count;

          break;
        }
      }

      // indices
      {
        const cgltf_accessor* accessor = gltfPrim->indices;

        uint32_t* writeIndices = (uint32_t*)(geom.triangles.data() + offsetTriangles);

        if(accessor->component_type == cgltf_component_type_r_32u && accessor->type == cgltf_type_scalar
           && accessor->stride == sizeof(uint32_t))
        {
          memcpy(writeIndices, cgltf_buffer_view_data(accessor->buffer_view) + accessor->offset,
                 sizeof(uint32_t) * accessor->count);
        }
        else
        {
          for(size_t i = 0; i < accessor->count; i++)
          {
            writeIndices[i] = (uint32_t)cgltf_accessor_read_index(gltfPrim->indices, i);
          }
        }

        offsetTriangles += (uint32_t)accessor->count / 3;
      }
    }
  }

  if(data->scenes_count > 0)
  {
    const cgltf_scene scene = (data->scene != nullptr) ? (*(data->scene)) : (data->scenes[0]);
    for(size_t nodeIdx = 0; nodeIdx < scene.nodes_count; nodeIdx++)
    {
      addInstancesFromNode(m_instances, data.get(), scene.nodes[nodeIdx]);
    }
  }
  else
  {
    for(size_t nodeIdx = 0; nodeIdx < data->nodes_count; nodeIdx++)
    {
      if(data->nodes[nodeIdx].parent == nullptr)
      {
        addInstancesFromNode(m_instances, data.get(), &(data->nodes[nodeIdx]));
      }
    }
  }

  return true;
}
}  // namespace animatedclusters
