//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2020 Apple Inc. All Rights Reserved.
//

// originally taken (and adapted) from https://github.com/ingowald/optix7course, 
// under following license

// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>

extern "C" char embedded_ptx_code[];

/*! SBT record for a raygen program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  RaySBTDataBuf data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void *data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  TriangleMeshSBTDataBuf data;
};

//! add aligned cube with front-lower-left corner and size
void TriangleMesh::addCube(const vec3f &center, const vec3f &size)
{
  affine3f xfm;
  xfm.p = center - 0.5f*size;
  xfm.l.vx = vec3f(size.x,0.f,0.f);
  xfm.l.vy = vec3f(0.f,size.y,0.f);
  xfm.l.vz = vec3f(0.f,0.f,size.z);
  addUnitCube(xfm);
}

/*! add a unit cube (subject to given xfm matrix) to the current
    triangleMesh */
void TriangleMesh::addUnitCube(const affine3f &xfm)
{
  int firstVertexID = (int)vertex.size();
  vertex.emplace_back(xfmPoint(xfm,vec3f(0.f,0.f,0.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(1.f,0.f,0.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(0.f,1.f,0.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(1.f,1.f,0.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(0.f,0.f,1.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(1.f,0.f,1.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(0.f,1.f,1.f)));
  vertex.emplace_back(xfmPoint(xfm,vec3f(1.f,1.f,1.f)));


  int indices[] = {0,1,3, 2,3,0,
                    5,7,6, 5,6,4,
                    0,4,5, 0,5,1,
                    2,3,7, 2,7,6,
                    1,5,7, 1,7,3,
                    4,0,2, 4,2,6
                    };
  for (int i=0;i<12;i++)
    index.emplace_back(firstVertexID+vec3i(indices[3*i+0],
                                        indices[3*i+1],
                                        indices[3*i+2]));
}

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
SampleRenderer::SampleRenderer(const std::vector<TriangleMeshBuf> &meshes, const int deviceID)
{
  initOptix();
    
  std::cout << "[PyRT] creating optix context ..." << std::endl;
  createContext(deviceID);
    
  std::cout << "[PyRT] setting up module ..." << std::endl;
  createModule();

  std::cout << "[PyRT] creating raygen programs ..." << std::endl;
  createRaygenPrograms();
  std::cout << "[PyRT] creating miss programs ..." << std::endl;
  createMissPrograms();
  std::cout << "[PyRT] creating hitgroup programs ..." << std::endl;
  createHitgroupPrograms();

  launchParams.traversable = buildAccel(meshes);
  
  std::cout << "[PyRT] setting up optix pipeline ..." << std::endl;
  createPipeline();

  std::cout << "[PyRT] building SBT ..." << std::endl;
  buildSBT(meshes);

  launchParamsBuffer.alloc(sizeof(launchParams));
  std::cout << "[PyRT] context, module, pipeline, etc, all set up ..." << std::endl;

  std::cout << GDT_TERMINAL_GREEN;
  std::cout << "[PyRT] Optix 7 Sample fully set up";
  std::cout << GDT_TERMINAL_DEFAULT << std::endl;
}

OptixTraversableHandle SampleRenderer::buildAccel(const std::vector<TriangleMeshBuf> &meshes)
{
  // meshes.resize(1);

  vertexBuffer.resize(meshes.size());
  indexBuffer.resize(meshes.size());
  
  OptixTraversableHandle asHandle { 0 };
  
  // ==================================================================
  // triangle inputs
  // ==================================================================
  std::vector<OptixBuildInput> triangleInput(meshes.size());
  std::vector<CUdeviceptr> d_vertices(meshes.size());
  std::vector<CUdeviceptr> d_indices(meshes.size());
  std::vector<uint32_t> triangleInputFlags(meshes.size());

  for (int meshID=0;meshID<meshes.size();meshID++) {
    // upload the model to the device: the builder
    const TriangleMeshBuf &model = meshes[meshID];
    vertexBuffer[meshID].alloc_and_upload_buf(model.vertex, 3*model.num_vertex);
    indexBuffer[meshID].alloc_and_upload_buf(model.index, 3*model.num_index);

    triangleInput[meshID] = {};
    triangleInput[meshID].type
      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
    d_indices[meshID]  = indexBuffer[meshID].d_pointer();
      
    triangleInput[meshID].triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(float)*3;
    triangleInput[meshID].triangleArray.numVertices         = (int)model.num_vertex;
    triangleInput[meshID].triangleArray.vertexBuffers       = &d_vertices[meshID];
    
    triangleInput[meshID].triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput[meshID].triangleArray.indexStrideInBytes  = sizeof(uint32_t)*3;
    triangleInput[meshID].triangleArray.numIndexTriplets    = (int)model.num_index;
    triangleInput[meshID].triangleArray.indexBuffer         = d_indices[meshID];
    
    triangleInputFlags[meshID] = 0 ;
    
    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput[meshID].triangleArray.flags               = &triangleInputFlags[meshID];
    triangleInput[meshID].triangleArray.numSbtRecords               = 1;
    triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer        = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
    triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0; 
  }
  // ==================================================================
  // BLAS setup
  // ==================================================================
  
  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE
    | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
    ;
  accelOptions.motionOptions.numKeys  = 1;
  accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
  
  OptixAccelBufferSizes blasBufferSizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage
              (optixContext,
                &accelOptions,
                triangleInput.data(),
                (int)meshes.size(),  // num_build_inputs
                &blasBufferSizes
                ));
  
  // ==================================================================
  // prepare compaction
  // ==================================================================
  
  CUDABuffer compactedSizeBuffer;
  compactedSizeBuffer.alloc(sizeof(uint64_t));
  
  OptixAccelEmitDesc emitDesc;
  emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = compactedSizeBuffer.d_pointer();
  
  // ==================================================================
  // execute build (main stage)
  // ==================================================================
  
  CUDABuffer tempBuffer;
  tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
  
  CUDABuffer outputBuffer;
  outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
    
  OPTIX_CHECK(optixAccelBuild(optixContext,
                              /* stream */0,
                              &accelOptions,
                              triangleInput.data(),
                              (int)meshes.size(),
                              tempBuffer.d_pointer(),
                              tempBuffer.sizeInBytes,
                              
                              outputBuffer.d_pointer(),
                              outputBuffer.sizeInBytes,
                              
                              &asHandle,
                              
                              &emitDesc,1
                              ));
  CUDA_SYNC_CHECK();
  
  // ==================================================================
  // perform compaction
  // ==================================================================
  uint64_t compactedSize;
  compactedSizeBuffer.download(&compactedSize,1);
  
  asBuffer.alloc(compactedSize);
  OPTIX_CHECK(optixAccelCompact(optixContext,
                                /*stream:*/0,
                                asHandle,
                                asBuffer.d_pointer(),
                                asBuffer.sizeInBytes,
                                &asHandle));
  CUDA_SYNC_CHECK();
  
  // ==================================================================
  // aaaaaand .... clean up
  // ==================================================================
  outputBuffer.free(); // << the UNcompacted, temporary output buffer
  tempBuffer.free();
  compactedSizeBuffer.free();
  
  return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void SampleRenderer::initOptix()
{
  std::cout << "[PyRT] initializing optix..." << std::endl;
    
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("[PyRT] no CUDA capable devices found!");
  std::cout << "[PyRT] found " << numDevices << " CUDA devices" << std::endl;

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK( optixInit() );
  std::cout << GDT_TERMINAL_GREEN
            << "[PyRT] successfully initialized optix... yay!"
            << GDT_TERMINAL_DEFAULT << std::endl;
}

static void context_log_cb(unsigned int level,
                            const char *tag,
                            const char *message,
                            void *)
{
  fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

/*! creates and configures a optix device context (in this simple
    example, only for the primary GPU device) */
void SampleRenderer::createContext(const int deviceID)
{
  // for this sample, do everything on one device
  // const int deviceID = 0;
  CUDA_CHECK(SetDevice(deviceID));
  CUDA_CHECK(StreamCreate(&stream));
    
  cudaGetDeviceProperties(&deviceProps, deviceID);
  std::cout << "[PyRT] running on device: " << deviceProps.name << std::endl;
    
  CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
  if( cuRes != CUDA_SUCCESS ) 
    fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
    
  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
  OPTIX_CHECK(optixDeviceContextSetLogCallback
              (optixContext,context_log_cb,nullptr,4));
}



/*! creates the module that contains all the programs we are going
    to use. in this simple example, we use a single module from a
    single .cu file, using a single embedded ptx string */
void SampleRenderer::createModule()
{
  moduleCompileOptions.maxRegisterCount  = 50;
  moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions = {};
  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur     = false;
  pipelineCompileOptions.numPayloadValues   = 2;
  pipelineCompileOptions.numAttributeValues = 2;
  pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
    
  pipelineLinkOptions.maxTraceDepth          = 2;
    
  const std::string ptxCode = embedded_ptx_code;
    
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                        &moduleCompileOptions,
                                        &pipelineCompileOptions,
                                        ptxCode.c_str(),
                                        ptxCode.size(),
                                        log,&sizeof_log,
                                        &module
                                        ));
  if (sizeof_log > 1) PRINT(log);
}
  


/*! does all setup for the raygen program(s) we are going to use */
void SampleRenderer::createRaygenPrograms()
{
  // we do a single ray gen program in this example:
  raygenPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module            = module;           
  pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &raygenPGs[0]
                                      ));
  if (sizeof_log > 1) PRINT(log);
}
  
/*! does all setup for the miss program(s) we are going to use */
void SampleRenderer::createMissPrograms()
{
  // we do a single ray gen program in this example:
  missPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module            = module;           
  pgDesc.miss.entryFunctionName = "__miss__radiance";

  // OptixProgramGroup raypg;
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &missPGs[0]
                                      ));
  if (sizeof_log > 1) PRINT(log);
}
  
/*! does all setup for the hitgroup program(s) we are going to use */
void SampleRenderer::createHitgroupPrograms()
{
  // for this simple example, we set up a single hit group
  hitgroupPGs.resize(1);
    
  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc pgDesc    = {};
  pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDesc.hitgroup.moduleCH            = module;           
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgDesc.hitgroup.moduleAH            = module;           
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                      &pgDesc,
                                      1,
                                      &pgOptions,
                                      log,&sizeof_log,
                                      &hitgroupPGs[0]
                                      ));
  if (sizeof_log > 1) PRINT(log);
}
  

/*! assembles the full pipeline of all programs */
void SampleRenderer::createPipeline()
{
  std::vector<OptixProgramGroup> programGroups;
  for (auto pg : raygenPGs)
    programGroups.emplace_back(pg);
  for (auto pg : missPGs)
    programGroups.emplace_back(pg);
  for (auto pg : hitgroupPGs)
    programGroups.emplace_back(pg);
    
  char log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK(optixPipelineCreate(optixContext,
                                  &pipelineCompileOptions,
                                  &pipelineLinkOptions,
                                  programGroups.data(),
                                  (int)programGroups.size(),
                                  log,&sizeof_log,
                                  &pipeline
                                  ));
  if (sizeof_log > 1) PRINT(log);

  OPTIX_CHECK(optixPipelineSetStackSize
              (/* [in] The pipeline to configure the stack size for */
                pipeline, 
                /* [in] The direct stack size requirement for direct
                  callables invoked from IS or AH. */
                2*1024,
                /* [in] The direct stack size requirement for direct
                  callables invoked from RG, MS, or CH.  */                 
                2*1024,
                /* [in] The continuation stack requirement. */
                2*1024,
                /* [in] The maximum depth of a traversable graph
                  passed to trace. */
                1));
  if (sizeof_log > 1) PRINT(log);
}

void SampleRenderer::setRaySBT(const RayBuf& ray_) {
  rayOriginBuffer.free();
  rayDirBuffer.free();
  rayOriginBuffer.alloc_and_upload_buf(ray_.origin, 3*ray_.num);
  rayDirBuffer.alloc_and_upload_buf(ray_.dir, 3*ray_.num);
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (int i=0;i<1;i++) {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
    rec.data.origin = (float*)rayOriginBuffer.d_pointer();
    rec.data.dir = (float*)rayDirBuffer.d_pointer();
    raygenRecords.emplace_back(rec);
  }
  // raygenRecordsBuffer.alloc_and_upload(raygenRecords);
  raygenRecordsBuffer.upload((const RaygenRecord*)raygenRecords.data(),raygenRecords.size());
  // sbt.raygenRecord = raygenRecordsBuffer.d_pointer();
}

void SampleRenderer::setRaySBT_torch(float *origin, float *dir) {
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for (int i=0;i<1;i++) {
    RaygenRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
    rec.data.origin = origin;
    rec.data.dir = dir;
    raygenRecords.emplace_back(rec);
  }
  // raygenRecordsBuffer.alloc_and_upload(raygenRecords);
  raygenRecordsBuffer.upload((const RaygenRecord*)raygenRecords.data(),raygenRecords.size());
  // sbt.raygenRecord = raygenRecordsBuffer.d_pointer();
}

/*! constructs the shader binding table */
void SampleRenderer::buildSBT(const std::vector<TriangleMeshBuf> &meshes)
{
  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for (int i=0;i<missPGs.size();i++) {
    MissRecord rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
    rec.data = nullptr; /* for now ... */
    missRecords.emplace_back(rec);
  }
  missRecordsBuffer.alloc_and_upload(missRecords);
  sbt.missRecordBase          = missRecordsBuffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof(MissRecord);
  sbt.missRecordCount         = (int)missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
  int numObjects = (int)meshes.size();
  std::vector<HitgroupRecord> hitgroupRecords;
  for (int meshID=0;meshID<numObjects;meshID++) {
    HitgroupRecord rec;
    // all meshes use the same code, so all same hit group
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0],&rec));
    // rec.data.color  = meshes[meshID].color;
    rec.data.vertex = (float*)vertexBuffer[meshID].d_pointer();
    rec.data.index  = (uint32_t*)indexBuffer[meshID].d_pointer();
    hitgroupRecords.emplace_back(rec);
  }
  hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
  sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
  sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  raygenRecordsBuffer.alloc(1*sizeof(RaygenRecord));
  sbt.raygenRecord = raygenRecordsBuffer.d_pointer();
}



/*! render one frame */
void SampleRenderer::render()
{
  // sanity check: make sure we launch only after first resize is
  // already done:
  if (launchParams.frame.size.x == 0) return;
    
  launchParamsBuffer.upload(&launchParams,1);
    
  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipeline,stream,
                          /*! parameters and SBT */
                          launchParamsBuffer.d_pointer(),
                          launchParamsBuffer.sizeInBytes,
                          &sbt,
                          /*! dimensions of the launch: */
                          launchParams.frame.size.x,
                          launchParams.frame.size.y,
                          1
                          ));
  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  CUDA_SYNC_CHECK();
}

/*! set camera to render with */
void SampleRenderer::setCamera(const Camera &camera)
{
  lastSetCamera = camera;
  launchParams.camera.position  = camera.from;
  launchParams.camera.direction = normalize(camera.at-camera.from);
  const float cosFovy = 0.66f;
  const float aspect = launchParams.frame.size.x / float(launchParams.frame.size.y);
  launchParams.camera.horizontal
    = cosFovy * aspect * normalize(cross(launchParams.camera.direction,
                                          camera.up));
  launchParams.camera.vertical
    = cosFovy * normalize(cross(launchParams.camera.horizontal,
                                launchParams.camera.direction));
}

/*! resize frame buffer to given resolution */
void SampleRenderer::resize(const vec2i &newSize)
{
  // if window minimized
  if (newSize.x == 0 | newSize.y == 0) return;
  
  // resize our cuda frame buffer
  colorBuffer.resize(newSize.x*newSize.y*6*sizeof(float));

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.frame.size  = newSize;
  launchParams.frame.colorBuffer = (float*)colorBuffer.d_pointer();

  // and re-set the camera, since aspect may have changed
  setCamera(lastSetCamera);
}

void SampleRenderer::resize_torch(const vec2i &newSize, float *out_buf)
{
  // if window minimized
  if (newSize.x == 0 | newSize.y == 0) return;

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.frame.size  = newSize;
  launchParams.frame.colorBuffer = out_buf;

  // and re-set the camera, since aspect may have changed
  setCamera(lastSetCamera);
}

/*! download the rendered color buffer */
void SampleRenderer::downloadPixels(float h_pixels[])
{
  colorBuffer.download(h_pixels,
                        launchParams.frame.size.x*launchParams.frame.size.y*6);
}
